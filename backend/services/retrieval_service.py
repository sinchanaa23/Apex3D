"""
Reference Retrieval Service for Apex3D.
Searches Wikimedia Commons API for openly licensed vehicle reference images,
captures CC BY-SA / CC0 metadata, deduplicates via perceptual hashing,
filters non-road/unrelated cars, and categorizes into standard viewpoints.
"""

import re
import uuid
import logging
import urllib.parse
from io import BytesIO
from typing import List, Dict, Optional, Any, Tuple
import requests
from PIL import Image
import imagehash

from backend.config import (
    WIKIMEDIA_API_URL,
    USER_AGENT,
    REFERENCES_DIR,
)
from backend.data.stanford_cars_classes import get_class_info
from backend.data.sample_references import get_sample_references_for_query
from backend.models import ReferenceCandidate, ReferencesResponse
from backend.database import save_cached_reference, get_cached_references

logger = logging.getLogger(__name__)

# Permissive open licenses accepted
PERMISSIVE_LICENSES = {
    "cc0",
    "public domain",
    "pd",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}

# Negative keywords to filter out non-car / non-exterior photos
NEGATIVE_KEYWORDS = [
    "interior",
    "engine",
    "dashboard",
    "steering wheel",
    "wheel",
    "badge",
    "logo",
    "crash",
    "scale model",
    "diecast",
    "toy",
]

VIEWPOINTS = [
    "front",
    "front_three_quarter",
    "side",
    "rear",
    "rear_three_quarter",
]


class ReferenceRetrievalService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ReferenceRetrievalService()
        return cls._instance

    def retrieve(
        self,
        class_id: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 12,
    ) -> ReferencesResponse:
        """
        Retrieves reference images for a vehicle class or query string.
        """
        class_info = get_class_info(class_id) if class_id is not None else None
        class_name = class_info["full_name"] if class_info else None

        # Build clean search query
        if query:
            search_query = query.strip()
        elif class_info:
            # Query make + model + year
            search_query = f"{class_info['make']} {class_info['model']} {class_info['year']}"
        else:
            search_query = "Vehicle"

        # Check local cache first
        cached = get_cached_references(class_id, limit=limit)
        if len(cached) >= 4:
            candidates = [ReferenceCandidate(**c) for c in cached]
            return self._build_response(search_query, class_id, class_name, candidates)

        # Fetch candidates from Wikimedia Commons API
        raw_candidates = self._fetch_wikimedia(search_query, class_info, limit=limit * 2)

        # Fallback to local reference database if network or API returns no results
        if not raw_candidates:
            logger.info("Wikimedia API returned 0 candidates (offline mode / network timeout). Using local reference fallback.")
            raw_candidates = get_sample_references_for_query(class_id=class_id, query=search_query, limit=limit)

        # Perceptual hash deduplication
        deduped = self._deduplicate_by_phash(raw_candidates)

        # Classify viewpoint and rank
        processed_candidates = []
        for cand in deduped[:limit]:
            cand["viewpoint"], cand["viewpoint_confidence"] = self._infer_viewpoint(
                cand["title"], cand.get("width"), cand.get("height")
            )
            # Cache in SQLite
            try:
                save_cached_reference(cand)
            except Exception as e:
                logger.debug("Failed caching reference: %s", e)

            processed_candidates.append(ReferenceCandidate(**cand))

        return self._build_response(search_query, class_id, class_name, processed_candidates)

    def _fetch_wikimedia(
        self, query: str, class_info: Optional[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Queries Wikimedia Commons API and parses response."""
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": "6",
            "gsrlimit": str(min(limit, 30)),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": "500",
        }
        headers = {"User-Agent": USER_AGENT}

        candidates = []
        try:
            resp = requests.get(WIKIMEDIA_API_URL, params=params, headers=headers, timeout=8)
            if resp.status_code != 200:
                logger.warning("Wikimedia API HTTP %s", resp.status_code)
                return candidates

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                title = page.get("title", "")
                title_lower = title.lower()

                # Filter negative keywords
                if any(neg in title_lower for neg in NEGATIVE_KEYWORDS):
                    continue

                imageinfo = page.get("imageinfo")
                if not imageinfo:
                    continue

                info = imageinfo[0]
                image_url = info.get("url")
                thumb_url = info.get("thumburl", image_url)
                width = info.get("width", 800)
                height = info.get("height", 600)

                if not image_url:
                    continue

                extmeta = info.get("extmetadata", {})
                lic_short = extmeta.get("LicenseShortName", {}).get("value", "CC BY-SA")
                lic_url = extmeta.get("LicenseUrl", {}).get("value")
                artist = extmeta.get("Artist", {}).get("value", "Wikimedia Contributor")
                # Clean html tags from artist
                clean_artist = re.sub(r"<[^>]*>", "", artist).strip()[:100]

                # Match scoring
                match_score = 0.85
                is_approx = False
                if class_info:
                    make_m = class_info["make"].lower() in title_lower
                    model_m = class_info["model"].lower() in title_lower
                    year_m = str(class_info["year"]) in title_lower
                    if make_m and model_m:
                        match_score = 0.95 if year_m else 0.85
                    elif make_m:
                        match_score = 0.65
                        is_approx = True
                    else:
                        match_score = 0.50
                        is_approx = True

                cand_id = f"wm_{page_id}"
                candidates.append({
                    "id": cand_id,
                    "class_id": class_info["class_id"] if class_info else None,
                    "title": title,
                    "image_url": image_url,
                    "thumb_url": thumb_url,
                    "source": "wikimedia_commons",
                    "license": lic_short,
                    "license_url": lic_url,
                    "author": clean_artist or "Unknown",
                    "width": width,
                    "height": height,
                    "match_score": match_score,
                    "is_approximate": is_approx,
                })

        except Exception as e:
            logger.warning("Error fetching Wikimedia references: %s", e)

        # Sort by match score descending
        candidates.sort(key=lambda c: c["match_score"], reverse=True)
        return candidates

    def _deduplicate_by_phash(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates candidate images using perceptual hashing on their thumbnails.
        Removes near-duplicate angles or duplicate uploads.
        """
        seen_hashes = []
        unique_candidates = []

        headers = {"User-Agent": USER_AGENT}
        for cand in candidates:
            thumb_url = cand.get("thumb_url")
            if not thumb_url:
                unique_candidates.append(cand)
                continue

            try:
                # Fast perceptual check on thumbnail
                resp = requests.get(thumb_url, headers=headers, timeout=4)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content))
                    p_hash = imagehash.phash(img)

                    # Compare against seen hashes
                    is_duplicate = False
                    for existing_hash in seen_hashes:
                        if (p_hash - existing_hash) <= 4:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        seen_hashes.append(p_hash)
                        unique_candidates.append(cand)
                else:
                    unique_candidates.append(cand)
            except Exception:
                unique_candidates.append(cand)

        return unique_candidates

    def _infer_viewpoint(
        self, title: str, width: Optional[int], height: Optional[int]
    ) -> Tuple[str, float]:
        """
        Infers viewpoint (front, front_three_quarter, side, rear, rear_three_quarter)
        from metadata keywords and aspect ratio analysis.
        """
        t = title.lower()

        if "front 3/4" in t or "front three quarter" in t or "front-left" in t or "front-right" in t:
            return "front_three_quarter", 0.92
        if "rear 3/4" in t or "rear three quarter" in t or "back-left" in t or "back-right" in t:
            return "rear_three_quarter", 0.90
        if "profile" in t or "side" in t or "lateral" in t:
            return "side", 0.95
        if "rear" in t or "back" in t or "tail" in t:
            return "rear", 0.90
        if "front" in t or "grille" in t or "fascia" in t:
            return "front", 0.90

        # Geometric aspect ratio heuristic
        if width and height and height > 0:
            aspect = float(width) / float(height)
            if aspect > 1.9:
                return "side", 0.70
            elif aspect > 1.4:
                return "front_three_quarter", 0.65
            elif 0.9 <= aspect <= 1.3:
                return "front", 0.60

        return "unknown", 0.0

    def _build_response(
        self,
        query: str,
        class_id: Optional[int],
        class_name: Optional[str],
        candidates: List[ReferenceCandidate],
    ) -> ReferencesResponse:
        """Constructs ReferencesResponse with viewpoint groupings."""
        groups: Dict[str, List[str]] = {v: [] for v in VIEWPOINTS}
        for cand in candidates:
            vp = cand.viewpoint
            if vp in groups:
                groups[vp].append(cand.id)

        return ReferencesResponse(
            success=True,
            query=query,
            class_id=class_id,
            class_name=class_name,
            total_candidates=len(candidates),
            candidates=candidates,
            viewpoint_groups=groups,
        )

    def save_manual_upload(
        self,
        image_bytes: bytes,
        filename: str,
        viewpoint: str = "unknown",
        class_id: Optional[int] = None,
        label: Optional[str] = None,
    ) -> ReferenceCandidate:
        """Saves a user-uploaded reference photo and indexes it."""
        ref_id = f"upload_{uuid.uuid4().hex[:8]}"
        ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
        save_name = f"{ref_id}.{ext}"
        save_path = REFERENCES_DIR / save_name

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        img = Image.open(BytesIO(image_bytes))
        w, h = img.size

        cand_data = {
            "id": ref_id,
            "class_id": class_id,
            "title": label or filename,
            "image_url": f"/api/references/file/{save_name}",
            "thumb_url": f"/api/references/file/{save_name}",
            "source": "manual_upload",
            "license": "User Contributed (Personal)",
            "license_url": None,
            "author": "Current User",
            "viewpoint": viewpoint if viewpoint in VIEWPOINTS else "unknown",
            "viewpoint_confidence": 1.0 if viewpoint in VIEWPOINTS else 0.0,
            "match_score": 1.0,
            "is_approximate": False,
            "width": w,
            "height": h,
        }

        save_cached_reference(cand_data)
        return ReferenceCandidate(**cand_data)


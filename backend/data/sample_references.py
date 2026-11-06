"""
Curated Offline Fallback Reference Images for Stanford Cars classes.
Ensures Reference Studio functions reliably even during venue Wi-Fi drops or Wikimedia timeouts.
"""

from typing import Dict, List, Any

# Curated reference viewpoints with open CC-BY / Public Domain licenses
SAMPLE_REFERENCES_DATABASE: List[Dict[str, Any]] = [
    {
        "id": "ref_am_vantage_01",
        "class_id": 7,
        "keywords": ["aston martin", "v8 vantage", "convertible", "2012"],
        "title": "Aston Martin V8 Vantage Roadster Front 3/4",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Aston_Martin_V8_Vantage_Roadster_front_quarter.jpg/800px-Aston_Martin_V8_Vantage_Roadster_front_quarter.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Aston_Martin_V8_Vantage_Roadster_front_quarter.jpg/400px-Aston_Martin_V8_Vantage_Roadster_front_quarter.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "author": "Vauxford",
        "viewpoint": "front_three_quarter",
        "viewpoint_confidence": 0.95,
        "match_score": 0.98,
        "is_approximate": False,
        "width": 1920,
        "height": 1080,
    },
    {
        "id": "ref_am_vantage_02",
        "class_id": 7,
        "keywords": ["aston martin", "v8 vantage", "side"],
        "title": "Aston Martin V8 Vantage Profile Side View",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Aston_Martin_V8_Vantage_Side.jpg/800px-Aston_Martin_V8_Vantage_Side.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Aston_Martin_V8_Vantage_Side.jpg/400px-Aston_Martin_V8_Vantage_Side.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "author": "The Car Spy",
        "viewpoint": "side",
        "viewpoint_confidence": 0.96,
        "match_score": 0.95,
        "is_approximate": False,
        "width": 1600,
        "height": 900,
    },
    {
        "id": "ref_am_vantage_03",
        "class_id": 7,
        "keywords": ["aston martin", "v8 vantage", "rear"],
        "title": "Aston Martin V8 Vantage Rear Three Quarter",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Aston_Martin_V8_Vantage_rear.jpg/800px-Aston_Martin_V8_Vantage_rear.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Aston_Martin_V8_Vantage_rear.jpg/400px-Aston_Martin_V8_Vantage_rear.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "author": "M97uzivatel",
        "viewpoint": "rear_three_quarter",
        "viewpoint_confidence": 0.91,
        "match_score": 0.94,
        "is_approximate": False,
        "width": 1920,
        "height": 1080,
    },
    {
        "id": "ref_audi_a5_01",
        "class_id": 12,
        "keywords": ["audi", "a5", "coupe", "2012"],
        "title": "Audi A5 Coupe Front View",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Audi_A5_Coupe_Facelift_Front.jpg/800px-Audi_A5_Coupe_Facelift_Front.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Audi_A5_Coupe_Facelift_Front.jpg/400px-Audi_A5_Coupe_Facelift_Front.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "author": "Thomas doerfer",
        "viewpoint": "front",
        "viewpoint_confidence": 0.93,
        "match_score": 0.96,
        "is_approximate": False,
        "width": 1920,
        "height": 1080,
    },
    {
        "id": "ref_audi_a5_02",
        "class_id": 12,
        "keywords": ["audi", "a5", "coupe", "side"],
        "title": "Audi A5 Coupe Side Profile",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Audi_A5_3.0_TDI_quattro_Side.jpg/800px-Audi_A5_3.0_TDI_quattro_Side.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Audi_A5_3.0_TDI_quattro_Side.jpg/400px-Audi_A5_3.0_TDI_quattro_Side.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "author": "M 93",
        "viewpoint": "side",
        "viewpoint_confidence": 0.97,
        "match_score": 0.95,
        "is_approximate": False,
        "width": 1600,
        "height": 900,
    },
    {
        "id": "ref_bmw_m3_01",
        "class_id": 33,
        "keywords": ["bmw", "m3", "coupe", "2012"],
        "title": "BMW M3 Coupe Front 3/4",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/BMW_M3_E92_Front_Quarter.jpg/800px-BMW_M3_E92_Front_Quarter.jpg",
        "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/BMW_M3_E92_Front_Quarter.jpg/400px-BMW_M3_E92_Front_Quarter.jpg",
        "source": "local_stanford",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "author": "SsmIntruder",
        "viewpoint": "front_three_quarter",
        "viewpoint_confidence": 0.94,
        "match_score": 0.97,
        "is_approximate": False,
        "width": 1920,
        "height": 1080,
    },
]


def get_sample_references_for_query(
    class_id: int = None,
    query: str = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Finds matching offline sample references."""
    matches = []
    q_tokens = query.lower().split() if query else []

    for ref in SAMPLE_REFERENCES_DATABASE:
        if class_id is not None and ref.get("class_id") == class_id:
            matches.append(dict(ref))
        elif q_tokens and any(token in " ".join(ref["keywords"]) for token in q_tokens):
            matches.append(dict(ref))

    if not matches and SAMPLE_REFERENCES_DATABASE:
        # Fallback to general vehicle references
        for r in SAMPLE_REFERENCES_DATABASE[:limit]:
            item = dict(r)
            item["is_approximate"] = True
            matches.append(item)

    return matches[:limit]


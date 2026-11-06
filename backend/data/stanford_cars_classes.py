"""
Stanford Cars (Cars196) Dataset - 196 Fine-Grained Vehicle Classes.
Contains the official 196 class labels with parsing utilities to extract
Make, Model, Body Style, and Model Year.
"""

from typing import Dict, List, Optional, Any

# Official 196 classes ordered by class_id (0 to 195)
STANFORD_CARS_CLASSES: List[str] = [
    "AM General Hummer SUV 2000",
    "Acura RL Sedan 2012",
    "Acura TL Sedan 2012",
    "Acura TL Type-S 2008",
    "Acura TSX Sedan 2012",
    "Acura Integra Type R 2001",
    "Acura ZDX Hatchback 2012",
    "Aston Martin V8 Vantage Convertible 2012",
    "Aston Martin V8 Vantage Coupe 2012",
    "Aston Martin Virage Convertible 2012",
    "Aston Martin Virage Coupe 2012",
    "Audi RS 4 Convertible 2008",
    "Audi A5 Coupe 2012",
    "Audi TTS Coupe 2012",
    "Audi R8 Coupe 2012",
    "Audi V8 Sedan 1994",
    "Audi 100 Sedan 1994",
    "Audi 100 Wagon 1994",
    "Audi TT Hatchback 2011",
    "Audi S6 Sedan 2011",
    "Audi S5 Convertible 2012",
    "Audi S5 Coupe 2012",
    "Audi S4 Sedan 2012",
    "Audi S4 Sedan 2007",
    "Audi TT RS Coupe 2012",
    "BMW ActiveHybrid 5 Sedan 2012",
    "BMW 1 Series Convertible 2012",
    "BMW 1 Series Coupe 2012",
    "BMW 3 Series Sedan 2012",
    "BMW 3 Series Wagon 2012",
    "BMW 6 Series Convertible 2007",
    "BMW X5 SUV 2007",
    "BMW X6 SUV 2012",
    "BMW M3 Coupe 2012",
    "BMW M5 Sedan 2010",
    "BMW M6 Convertible 2010",
    "BMW X3 SUV 2012",
    "BMW Z4 Convertible 2012",
    "Bentley Continental Supersports Conv. Convertible 2012",
    "Bentley Arnage Sedan 2009",
    "Bentley Mulsanne Sedan 2011",
    "Bentley Continental GT Coupe 2012",
    "Bentley Continental GT Coupe 2007",
    "Bentley Continental Flying Spur Sedan 2007",
    "Bugatti Veyron 16.4 Convertible 2009",
    "Bugatti Veyron 16.4 Coupe 2009",
    "Buick Regal GS 2012",
    "Buick Rainier SUV 2007",
    "Buick Verano Sedan 2012",
    "Buick Enclave SUV 2012",
    "Cadillac CTS-V Sedan 2012",
    "Cadillac SRX SUV 2012",
    "Cadillac Escalade EXT Crew Cab 2007",
    "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
    "Chevrolet Corvette Convertible 2012",
    "Chevrolet Corvette ZR1 2012",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007",
    "Chevrolet Traverse SUV 2012",
    "Chevrolet Camaro Convertible 2012",
    "Chevrolet HHR SS 2008",
    "Chevrolet Impala Sedan 2007",
    "Chevrolet Tahoe Hybrid SUV 2012",
    "Chevrolet Sonic Sedan 2012",
    "Chevrolet Express Cargo Van 2007",
    "Chevrolet Avalanche Crew Cab 2012",
    "Chevrolet Cobalt SS 2010",
    "Chevrolet Venture Extended Passenger 1999",
    "Chevrolet Silverado 1500 Regular Cab 2012",
    "Chevrolet Silverado 1500 Extended Cab 2012",
    "Chevrolet TrailBlazer SS 2009",
    "Chevrolet Silverado 2500HD Regular Cab 2012",
    "Chevrolet Express Van 2007",
    "Chrysler 300 SRT-8 2010",
    "Chrysler Mastercraft PT Cruiser Convertible 2008",
    "Chrysler PT Cruiser Convertible 2008",
    "Chrysler Sebring Convertible 2010",
    "Chrysler Town and Country Minivan 2012",
    "Chrysler Crossfire Convertible 2008",
    "Chrysler Aspen SUV 2009",
    "Daewoo Nubira Wagon 1999",
    "Dodge Caliber Wagon 2012",
    "Dodge Caliber Wagon 2007",
    "Dodge Caravan Minivan 1997",
    "Dodge Ram Pickup 3500 Crew Cab 2010",
    "Dodge Ram Pickup 3500 Quad Cab 2009",
    "Dodge Challenger SRT8 2011",
    "Dodge Charger Sedan 2012",
    "Dodge Charger SRT-8 2009",
    "Dodge Dakota Club Cab 2007",
    "Dodge Dakota Crew Cab 2010",
    "Dodge Durango SUV 2012",
    "Dodge Durango SUV 2004",
    "Dodge Journey SUV 2012",
    "Dodge Magnum Wagon 2008",
    "Eagle Talon Hatchback 1998",
    "FIAT 500 Abarth 2012",
    "FIAT 500 Convertible 2012",
    "Ferrari FF Coupe 2012",
    "Ferrari California Convertible 2012",
    "Ferrari 458 Italia Convertible 2012",
    "Ferrari 458 Italia Coupe 2012",
    "Fisker Karma Sedan 2012",
    "Ford F-450 Super Duty Crew Cab 2012",
    "Ford Mustang Convertible 2007",
    "Ford Fiesta Sedan 2012",
    "Ford Freestar Minivan 2007",
    "Ford Expedition EL SUV 2009",
    "Ford Edge SUV 2012",
    "Ford Ranger SuperCab 2011",
    "Ford GT Coupe 2006",
    "Ford F-150 Regular Cab 2012",
    "Ford F-150 Regular Cab 2007",
    "Ford Focus Sedan 2007",
    "Ford E-Series Wagon Van 2012",
    "GMC Terrain SUV 2012",
    "GMC Savana Van 2012",
    "GMC Yukon Hybrid SUV 2012",
    "GMC Acadia SUV 2012",
    "GMC Canyon Extended Cab 2012",
    "Geo Metro Convertible 1993",
    "HUMMER H3T Crew Cab 2010",
    "HUMMER H2 SUT Crew Cab 2009",
    "Honda Odyssey Minivan 2012",
    "Honda Odyssey Minivan 2007",
    "Honda Accord Coupe 2012",
    "Honda Accord Sedan 2012",
    "Hyundai Veloster Hatchback 2012",
    "Hyundai Santa Fe SUV 2012",
    "Hyundai Tucson SUV 2012",
    "Hyundai Veracruz SUV 2012",
    "Hyundai Sonata Hybrid Sedan 2012",
    "Hyundai Elantra Sedan 2007",
    "Hyundai Accent Sedan 2012",
    "Hyundai Genesis Sedan 2012",
    "Hyundai Sonata Sedan 2012",
    "Hyundai Azera Sedan 2012",
    "Infiniti G37 Convertible 2012",
    "Infiniti G37 Coupe 2008",
    "Infiniti QX56 SUV 2011",
    "Isuzu Ascender SUV 2008",
    "Jaguar XK XKR 2012",
    "Jeep Patriot SUV 2012",
    "Jeep Wrangler SUV 2012",
    "Jeep Liberty SUV 2012",
    "Jeep Grand Cherokee SUV 2012",
    "Jeep Compass SUV 2012",
    "Lamborghini Reventon Convertible 2008",
    "Lamborghini Aventador Coupe 2012",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012",
    "Lamborghini Diablo Coupe 2001",
    "Land Rover Range Rover SUV 2012",
    "Land Rover LR2 SUV 2012",
    "Lincoln Town Car Sedan 2007",
    "Lincoln Navigator SUV 2012",
    "Lincoln MKS Sedan 2012",
    "Lincoln MKZ Sedan 2012",
    "MINI Cooper Roadster Convertible 2012",
    "Maybach Landaulet Convertible 2012",
    "Mazda Tribute SUV 2011",
    "McLaren MP4-12C Coupe 2012",
    "Mercedes-Benz 300-Class Convertible 1993",
    "Mercedes-Benz C-Class Sedan 2012",
    "Mercedes-Benz SL-Class Coupe 2009",
    "Mercedes-Benz E-Class Sedan 2012",
    "Mercedes-Benz S-Class Sedan 2012",
    "Mercedes-Benz Sprinter Van 2012",
    "Mitsubishi Lancer Sedan 2012",
    "Nissan Leaf Hatchback 2012",
    "Nissan NV Passenger Van 2012",
    "Nissan Juke Hatchback 2012",
    "Nissan 240SX Coupe 1998",
    "Nissan GT-R Coupe 2012",
    "Plymouth Neon Coupe 1999",
    "Porsche Panamera Sedan 2012",
    "Ram C/V Cargo Van Minivan 2012",
    "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
    "Rolls-Royce Ghost Sedan 2012",
    "Rolls-Royce Phantom Sedan 2012",
    "Scion xD Hatchback 2012",
    "Spyker C8 Convertible 2009",
    "Spyker C8 Coupe 2009",
    "Suzuki Aerio Sedan 2007",
    "Suzuki Kizashi Sedan 2012",
    "Suzuki SX4 Hatchback 2012",
    "Suzuki SX4 Sedan 2012",
    "Tesla Model S Sedan 2012",
    "Toyota Sequoia SUV 2012",
    "Toyota Camry Sedan 2012",
    "Toyota Corolla Sedan 2012",
    "Toyota 4Runner SUV 2012",
    "Volkswagen Golf Hatchback 2012",
    "Volkswagen Golf Hatchback 1991",
    "Volkswagen Beetle Hatchback 2012",
    "Volvo C30 Hatchback 2012",
    "Volvo 240 Sedan 1993",
    "Volvo XC90 SUV 2007",
    "smart fortwo Convertible 2012",
]

# Person A's list is the order the trained model uses (docs/INTEGRATION_NOTES_A_TO_C.md). It replaces the list above.
import json as _json
from pathlib import Path as _Path

_CLASS_FILE = _Path(__file__).resolve().parents[2] / "results" / "class_names.json"
if _CLASS_FILE.exists():
    _names = [row["name"] for row in sorted(_json.loads(_CLASS_FILE.read_text(encoding="utf-8")), key=lambda r: r["class_id"])]
    if len(_names) == 196:
        STANFORD_CARS_CLASSES = _names

MULTI_WORD_MAKES = [
    "AM General",
    "Aston Martin",
    "Land Rover",
    "Mercedes-Benz",
    "Rolls-Royce",
]

BODY_STYLES = [
    "Sedan",
    "SUV",
    "Coupe",
    "Convertible",
    "Hatchback",
    "Van",
    "Wagon",
    "Minivan",
    "Extended Cab",
    "Regular Cab",
    "SuperCab",
    "Crew Cab",
    "Quad Cab",
    "Club Cab",
    "Passenger",
]


def parse_vehicle_label(label: str) -> Dict[str, Any]:
    """
    Parses a Stanford Cars label string into its constituent components:
    Make, Model, Body Style, and Model Year.
    
    Example:
        'Aston Martin V8 Vantage Convertible 2012' ->
        {
            'make': 'Aston Martin',
            'model': 'V8 Vantage',
            'body_style': 'Convertible',
            'year': 2012,
            'full_name': 'Aston Martin V8 Vantage Convertible 2012'
        }
    """
    parts = label.strip().split()
    if not parts:
        return {"make": "Unknown", "model": "Unknown", "body_style": "Vehicle", "year": 2000, "full_name": label}

    # Year is typically the last token
    try:
        year = int(parts[-1])
        parts_without_year = parts[:-1]
    except ValueError:
        year = 2000
        parts_without_year = parts

    # Multi-word make check
    make = parts_without_year[0]
    rest_idx = 1
    for mwm in MULTI_WORD_MAKES:
        mwm_parts = mwm.split()
        if len(parts_without_year) >= len(mwm_parts):
            if " ".join(parts_without_year[: len(mwm_parts)]).lower() == mwm.lower():
                make = mwm
                rest_idx = len(mwm_parts)
                break

    rest_parts = parts_without_year[rest_idx:]

    # Check for body style in the remaining tokens
    body_style = "Car"
    model_parts = []
    found_body = False

    # Check two-word body styles first (e.g., "Crew Cab")
    for i in range(len(rest_parts) - 1):
        two_word = f"{rest_parts[i]} {rest_parts[i+1]}"
        if two_word in BODY_STYLES:
            body_style = two_word
            model_parts = rest_parts[:i]
            found_body = True
            break

    if not found_body:
        # Check single-word body styles from right to left
        for i in range(len(rest_parts) - 1, -1, -1):
            if rest_parts[i] in BODY_STYLES:
                body_style = rest_parts[i]
                model_parts = rest_parts[:i]
                found_body = True
                break

    if not found_body:
        model_parts = rest_parts

    model = " ".join(model_parts) if model_parts else "Model"

    return {
        "make": make,
        "model": model,
        "body_style": body_style,
        "year": year,
        "full_name": label,
    }


def get_class_info(class_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns structured vehicle metadata for a given Stanford Cars class ID (0-195).
    """
    if 0 <= class_id < len(STANFORD_CARS_CLASSES):
        parsed = parse_vehicle_label(STANFORD_CARS_CLASSES[class_id])
        parsed["class_id"] = class_id
        return parsed
    return None


def search_classes(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Searches 196 Stanford Cars classes by keyword (make, model, or year).
    """
    query_lower = query.lower()
    matches = []
    for idx, name in enumerate(STANFORD_CARS_CLASSES):
        if query_lower in name.lower():
            info = get_class_info(idx)
            if info:
                matches.append(info)
        if len(matches) >= limit:
            break
    return matches


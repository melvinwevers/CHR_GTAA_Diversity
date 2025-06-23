#!/usr/bin/env python3
"""
data_processing.py: Ingest JSON photo metadata files, clean and normalise,
and save as Parquet for efficient analytics.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd
from pandas.api.types import is_object_dtype
from tqdm import tqdm

# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def normalize_name(name: Any) -> Any:
    """Strip honorifics and normalise capitalisation for person names."""
    if not isinstance(name, str):
        return name
        
    honorifics = {"dr", "mr", "mrs", "ms", "prof", "mevr", "dhr", "hr", "ir", "sign"}
    
    # Clean and split name
    name = name.strip()
    parts = name.split()
    
    # Remove honorific if present
    if parts and parts[0].lower().rstrip('.') in honorifics:
        parts = parts[1:]
        
    # Rejoin and clean spacing
    name = " ".join(parts).strip()
    name = re.sub(r'\s+', ' ', name)
    
    # Capitalize each word
    words = name.split()
    words = [w.upper() if w.replace('.','').isupper() else w.capitalize() for w in words]
    
    return " ".join(words)


def ensure_list(x: Any) -> List[Any]:
    """Ensure a value is always returned as a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []


def serialise_nested_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list/dict columns to JSON strings so PyArrow can handle them."""

    def should_serialise(series: pd.Series) -> bool:
        if not is_object_dtype(series):
            return False
        return bool(series.apply(lambda v: isinstance(v, (list, dict))).any())

    def to_json_safe(v: Any) -> Any:
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    for col in df.columns:
        if should_serialise(df[col]):
            print(f"→ Serialising nested column '{col}' to JSON strings …")
            df[col] = df[col].map(to_json_safe)
    return df


def standardise_location(loc: str) -> str:
    """Very simple location normaliser placeholder."""
    return loc.strip().lower()


def clean_creator_name(creator: Any) -> str:
    """Clean creator names by removing URL prefixes and reversing name order."""
    if not isinstance(creator, str):
        return ""
    
    # Remove URL prefix
    if creator.startswith("https://archief.nl/id/vervaardiger/"):
        creator = creator.replace("https://archief.nl/id/vervaardiger/", "")
    
    # Remove __anefo suffix if present
    if "__anefo" in creator:
        creator = creator.split("__anefo")[0]
    
    # Split by underscores to get name parts
    parts = creator.split("_")
    
    if len(parts) >= 2:
        # Original format: surname_firstname -> firstname surname
        firstname = parts[-1]  # last part is firstname
        surname_parts = parts[:-1]  # everything else is surname
        surname = " ".join(surname_parts)
        return f"{firstname} {surname}".lower()
    elif len(parts) == 1:
        return parts[0].lower()
    
    return ""

# ---------------------------------------------------------
#  JSON reader with encoding detection
# ---------------------------------------------------------

def read_json_file(file_path: Path) -> Any:
    raw = file_path.read_bytes()
    
    # Try common encodings first
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        pass
        
    # Fallback to latin-1 with error replacement
    return json.loads(raw.decode("latin-1", errors="replace"))


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

PREVIEW_COLS = [
    "photo_id",
    "date",
    "year",
    "decade",
    "quarter",
    "month",
    "has_image",
    "has_location",
    "has_subject",
    "has_person",
    "clean_description",
    "locations_standardized",
    "persons",
    "subjects_list",
    "creator",
    "isMateriaaltype",
    "format",
    "fotocollectie",
]


def main(json_dir: str | Path, output_path: str | Path) -> None:
    json_dir = Path(json_dir)
    output_path = Path(output_path)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    dfs: list[pd.DataFrame] = []

    file_iter: Iterable[Path] = tqdm(json_files, desc="Reading JSON", unit="file")

    for file_path in file_iter:
        try:
            data = read_json_file(file_path)
        except Exception as exc:
            print(
                f"! Skipping {file_path.name} – could not parse JSON ({exc})",
                file=sys.stderr,
            )
            continue
        dfs.append(pd.json_normalize(data, sep="_"))

    if not dfs:
        sys.exit("All files failed to parse; nothing to process.")

    df = pd.concat(dfs, ignore_index=True)

    # --------------------------------------
    # 1. Extract additional metadata fields
    # --------------------------------------
    # Map the fields from the JSON structure to our desired column names
    field_mappings = {
        "creator": "aggregation_metadata_http://purl.org/dc/terms/creator",
        "isMateriaaltype": "foto_metadata_https://archief.nl/def/ontologie/isMateriaaltype",
        "format": "foto_metadata_http://purl.org/dc/elements/1.1/format",
        "fotocollectie": "description_metadata_https://archief.nl/def/ontologie/titelFotocollectie",
    }

    for new_col, json_field in field_mappings.items():
        df[new_col] = df.get(json_field, pd.NA).astype(str).replace(['nan', 'None', 'NaN', 'null', '<NA>'], '').str.strip()
        # Convert empty strings after stripping to pd.NA for better filtering
        if new_col == 'fotocollectie':
            df[new_col] = df[new_col].replace('', pd.NA)

    # Clean up creator names specifically
    df['creator'] = df['creator'].apply(clean_creator_name)

    # Check if the image URL exists in the aggregation metadata
    image_url_field = "aggregation_metadata_http://www.europeana.eu/schemas/edm/isShownBy"
    df["has_image"] = df[image_url_field].notna() & (df[image_url_field] != "")

    # --------------------------------------
    # 2. Temporal fields
    # --------------------------------------
    date_col = "description_metadata_http://purl.org/dc/terms/date"
    df["date"] = pd.to_datetime(df.get(date_col), errors="coerce")
    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q")

    # --------------------------------------
    # 3. Locations
    # --------------------------------------
    loc_col = "description_metadata_https://archief.nl/def/ontologie/trefwoordLocatie"
    df["locations_list"] = df.get(loc_col).apply(ensure_list)
    df["has_location"] = df["locations_list"].apply(bool)
    
    # Clean locations more carefully
    def clean_locations_list(locations_list):
        """Clean locations list while avoiding empty/invalid entries"""
        if not locations_list:
            return []
        
        cleaned = []
        for location in locations_list:
            if pd.isna(location) or location is None:
                continue
            
            # Convert to string and clean
            location_str = str(location).strip()
            
            # Skip empty or meaningless locations
            if not location_str or location_str.lower() in ['', 'nan', 'none', 'null']:
                continue
            
            # Standardize the location
            standardized = standardise_location(location_str)
            
            if standardized:
                cleaned.append(standardized)
        
        return cleaned
    
    df["locations_list"] = df["locations_list"].apply(clean_locations_list)
    df["locations_standardized"] = df["locations_list"]  # rename for export

    # --------------------------------------
    # 4. Subjects
    # --------------------------------------
    subj_col = "description_metadata_https://archief.nl/def/ontologie/trefwoordAlgemeen"
    df["subjects_list"] = df.get(subj_col).apply(ensure_list)
    df["has_subject"] = df["subjects_list"].apply(bool)
    
    def clean_subjects_list(subjects_list):
        """Clean subjects list while preserving original vocabulary formatting"""
        if not subjects_list:
            return []
        
        cleaned = []
        for subject in subjects_list:
            if pd.isna(subject) or subject is None:
                continue
            
            subject_str = str(subject).strip()
            
            # Skip empty or meaningless subjects
            if not subject_str or subject_str.lower() in ['', 'nan', 'none', 'null']:
                continue
            
            # Handle specific compound corrupted case: 'beelhouwwerken",spandoeken'
            if 'beelhouwwerken",spandoeken' in subject_str:
                # Split this into two separate subjects
                cleaned.append('beeldhouwwerken')
                cleaned.append('spandoeken')
                continue
            
            # Handle pipe-separated subjects (like "bataljons |commando-overdrachten |generaal-majoors")
            if '|' in subject_str:
                pipe_subjects = subject_str.split('|')
                pipe_results = []
                for pipe_subject in pipe_subjects:
                    cleaned_pipe = clean_individual_subject(pipe_subject)
                    if cleaned_pipe:
                        cleaned.append(cleaned_pipe)
                        pipe_results.append(cleaned_pipe)
            else:
                cleaned_single = clean_individual_subject(subject_str)
                if cleaned_single:
                    cleaned.append(cleaned_single)
        
        return cleaned
    
    def clean_individual_subject(subject_str):
        """Clean an individual subject string"""
        if not subject_str:
            return None

        subject = subject_str.strip()
        
        # Specific hardcoded corrections for known corrupted subjects
        specific_corrections = {
            'beelhouwwerken"': 'beeldhouwwerken',
            '[rinsessen': 'prinsessen',
            '[aarden': 'paarden'
        }
        
        # Apply specific corrections first
        for corrupted, corrected in specific_corrections.items():
            if corrupted in subject:
                subject = subject.replace(corrupted, corrected)
        
        # Remove leading/trailing brackets, quotes, etc. (but keep apostrophes)
        subject = subject.strip('[](){}\"^`')  # Removed single quote '
        
        # Remove any remaining malformed characters at start/end
        while subject and subject[0] in '[("^`':  # Removed single quote '
            subject = subject[1:]
        while subject and subject[-1] in '])\"^`':  # Removed single quote '
            subject = subject[:-1]
        
        # Remove bracket characters from anywhere in the string
        subject = subject.replace('[', '').replace(']', '')
        

        subject = subject.strip()
        
        # Skip if now empty or invalid
        if not subject or subject.lower() in ['', 'nan', 'none', 'null']:
            return None
        
        subject_cleaned = ' '.join(subject.split()).lower()
        
        # Final validation - must have at least 2 characters and text
        if len(subject_cleaned) < 2 or subject_cleaned.isdigit():
            return None
            
        return subject_cleaned
    
    df["subjects_list"] = df["subjects_list"].apply(clean_subjects_list)

    # --------------------------------------
    # 5. Persons
    # --------------------------------------
    person_col = "description_metadata_https://archief.nl/def/ontologie/trefwoordPersoon"
    df["persons_list"] = df.get(person_col).apply(ensure_list)
    df["has_person"] = df["persons_list"].apply(bool)
    
    # Clean persons list more carefully
    def clean_persons_list(persons_list):
        """Clean persons list while avoiding empty/invalid entries"""
        if not persons_list:
            return []
        
        cleaned = []
        for person in persons_list:
            if pd.isna(person) or person is None:
                continue
            
            # Normalize the name
            cleaned_name = normalize_name(person)
            
            # Skip empty or meaningless names
            if not cleaned_name or cleaned_name.lower().strip() in ['', 'nan', 'none', 'null']:
                continue
            
            # Convert to lowercase for consistency but keep the normalized format
            cleaned.append(cleaned_name.lower())
        
        return cleaned
    
    df["persons_list"] = df["persons_list"].apply(clean_persons_list)
    df["persons"] = df["persons_list"]  # rename for export

    # --------------------------------------
    # 6. Description text
    # --------------------------------------
    desc_col = "description_metadata_http://purl.org/dc/elements/1.1/description"
    df["clean_description"] = (
        df.get(desc_col, "")
        .fillna("")
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )

    # Ensure photo_id exists – use source filename sans extension if missing
    if "photo_id" not in df.columns:
        df["photo_id"] = df.index.astype(str)

    # --------------------------------------
    # 6. Column subset & serialisation
    # --------------------------------------
    missing = [c for c in PREVIEW_COLS if c not in df.columns]
    if missing:
        print(f"! Warning: expected columns missing and filled with NA: {', '.join(missing)}")
        for c in missing:
            df[c] = pd.NA

    df_out = df[PREVIEW_COLS].copy()
    df_out = serialise_nested_values(df_out)


    print("\n📊 Data Quality Summary:")
    print(f"  • Total records: {len(df_out):,}")
    print(f"  • Records with images: {df_out['has_image'].sum():,} ({df_out['has_image'].mean():.1%})")
    print(f"  • Records with subjects: {df_out['has_subject'].sum():,} ({df_out['has_subject'].mean():.1%})")
    print(f"  • Records with persons: {df_out['has_person'].sum():,} ({df_out['has_person'].mean():.1%})")
    print(f"  • Records with locations: {df_out['has_location'].sum():,} ({df_out['has_location'].mean():.1%})")
    

    # --------------------------------------
    # 7. Write Parquet
    # --------------------------------------
    print("Writing cleaned data to Parquet – this may take a moment …")
    df_out.to_parquet(output_path, index=False)
    print(f"✓ Saved cleaned data to {output_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process photo metadata JSONs to Parquet")
    parser.add_argument("--input-dir", required=True, help="Directory of JSON files")
    parser.add_argument(
        "--output", default="data/photos_archive.parquet", help="Output Parquet file path"
    )

    args = parser.parse_args()
    try:
        main(args.input_dir, args.output)
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")

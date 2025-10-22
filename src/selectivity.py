import pandas as pd


def mask(filters: dict, metadata: pd.DataFrame):
    if metadata.empty:
        return 0.0
    
    mask = pd.Series(True, index=metadata.index)
    # filter metadata by each key and their condition
    for key, condition in filters.items():
        col = metadata[key]

        for op, val in condition.items():
            if op == "eq":
                mask &= col == val
            elif op == "ge":
                mask &= col >= val
            elif op == "le":
                mask &= col <= val
            elif op == "between":
                low, high = val
                mask &= col.between(low, high, inclusive="both")
            elif op == "in":
                mask &= col.isin(val)
            elif op == "like":
                mask &= col.astype(str).str.contains(val, case=False, na=False)
    return mask

"""
Provides fraction of metadata rows that pass given filters

"""
def compute_selectivity(filters: dict, metadata: pd.DataFrame) -> float:
    
    if filters == {}:
        return 1.0
    if metadata.empty:
        return 0.0
    
    mask = pd.Series(True, index=metadata.index)
    # filter metadata by each key and their condition
    for key, condition in filters.items():
        col = metadata[key]

        for op, val in condition.items():
            if op == "eq":
                mask &= col == val
            elif op == "ge":
                mask &= col >= val
            elif op == "le":
                mask &= col <= val
            elif op == "between":
                low, high = val
                mask &= col.between(low, high, inclusive="both")
            elif op == "in":
                mask &= col.isin(val)
            elif op == "like":
                mask &= col.astype(str).str.contains(val, case=False, na=False)
        num_passed = mask.sum()
        total = len(metadata)
        return float(num_passed / total) if total > 0 else 0.0
        

   
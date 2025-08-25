import pandas as pd

def load_keycode_map(keycode_file):
    """
    Load IRIS_KEYCODE file and return a mapping from DWHCode to KeyCode.
    """
    df_keycode = pd.read_csv(keycode_file)
    # Pivot so DWHCode is index, KeyCode is value
    keycode_map = pd.Series(df_keycode['KeyCode'].values, index=df_keycode['DWHCode']).to_dict()
    return keycode_map

def match_keycodes(data_file, keycode_map):
    """
    Replace column headers in data_file (e.g., IS_security, BS_security, Note_security)
    that match DWHCode with their corresponding KeyCode.
    Returns a DataFrame with updated column names.
    """
    df = pd.read_csv(data_file)
    new_columns = [keycode_map.get(col, col) for col in df.columns]
    df.columns = new_columns
    return df

if __name__ == "__main__":
    keycode_map = load_keycode_map('IRIS_KEYCODE.csv')
    df_is = match_keycodes('sql/IS_security.csv', keycode_map)
    df_bs = match_keycodes('sql/BS_security.csv', keycode_map)
    df_note = match_keycodes('sql/Note_security.csv', keycode_map)
    print('IS_security matched columns:')
    print(df_is.head())
    print('BS_security matched columns:')
    print(df_bs.head())
    print('Note_security matched columns:')
    print(df_note.head())
    print(keycode_map)
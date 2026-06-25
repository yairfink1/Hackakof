import joblib
from pathlib import Path

def uncompile_weights(input_path, output_path):
    print(f"Loading weights from {input_path}...")
    state_dict = joblib.load(input_path)
    
    new_state_dict = {}
    changed_keys = 0
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k.replace("_orig_mod.", "")
            changed_keys += 1
        else:
            new_key = k
        new_state_dict[new_key] = v
        
    print(f"Uncompiled {changed_keys} keys.")
    print(f"Saving uncompiled weights to {output_path}...")
    joblib.dump(new_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    team_dir = Path(__file__).resolve().parent
    input_weights = team_dir / "weights.joblib"
    output_weights = team_dir / "weights_uncompiled.joblib"
    
    if not input_weights.exists():
        print(f"Error: {input_weights} does not exist.")
    else:
        uncompile_weights(input_weights, output_weights)

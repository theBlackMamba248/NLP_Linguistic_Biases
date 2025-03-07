import json
import os

def load_existing_scenarios(filename="scenarios.json"):
    """
    Loads the existing scenarios from a JSON file.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        print(f"Loaded scenarios from {filename}")
        return scenarios
    else:
        print(f"{filename} not found!")
        return None

def main():
    scenarios = load_existing_scenarios("scenarios.json")
    if scenarios is not None:
        # Optionally print out the loaded scenarios for verification.
        import json
        print(json.dumps(scenarios, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()

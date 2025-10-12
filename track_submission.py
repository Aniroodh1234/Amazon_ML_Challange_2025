import json
import os
from datetime import datetime

TRACKER_FILE = '/content/Amazon_ML_Challange_2025/submission_log.json'

def log_submission(submission_num, config_params, model_type, val_smape, 
                   public_smape=None, notes="", output_file="outputs/test_out.csv"):
    """Log submission details"""
    
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            log = json.load(f)
    else:
        log = {"submissions": []}
    
    entry = {
        "submission_number": submission_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config_params,
        "model_type": model_type,
        "validation_smape": val_smape,
        "public_smape": public_smape,
        "notes": notes,
        "output_file": output_file,
        "status": "uploaded" if public_smape else "pending"
    }
    
    log["submissions"].append(entry)
    
    with open(TRACKER_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"Logged Submission #{submission_num}")
    print(f"Val SMAPE: {val_smape}%")
    if public_smape:
        print(f"Public SMAPE: {public_smape}%")

def update_public_smape(submission_num, public_smape):
    """Update public SMAPE after getting result"""
    with open(TRACKER_FILE, 'r') as f:
        log = json.load(f)
    
    for entry in log["submissions"]:
        if entry["submission_number"] == submission_num:
            entry["public_smape"] = public_smape
            entry["status"] = "uploaded"
            break
    
    with open(TRACKER_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"Updated #{submission_num}: Public SMAPE = {public_smape}%")

def view_submissions():
    """View all submissions"""
    if not os.path.exists(TRACKER_FILE):
        print("No submissions logged yet")
        return
    
    with open(TRACKER_FILE, 'r') as f:
        log = json.load(f)
    
    print("\n" + "="*70)
    print("SUBMISSION SUMMARY")
    print("="*70)
    
    for sub in log["submissions"]:
        print(f"\nSubmission #{sub['submission_number']}")
        print(f"   Time: {sub['timestamp']}")
        print(f"   Model: {sub['model_type']}")
        print(f"   Val SMAPE: {sub['validation_smape']}%")
        print(f"   Public SMAPE: {sub.get('public_smape', 'Pending')}%")
        print(f"   Notes: {sub['notes']}")
    
    # Show best
    completed = [s for s in log["submissions"] if s.get("public_smape")]
    if completed:
        best = min(completed, key=lambda x: x["public_smape"])
        print("\n" + "="*70)
        print(f"BEST: Submission #{best['submission_number']} - {best['public_smape']}%")
        print("="*70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        view_submissions()
    else:
        print("Submission tracker ready!")
        print("Usage: python track_submission.py view")


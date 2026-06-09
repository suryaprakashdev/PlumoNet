import requests
import sys

def test_inference(zip_path):
    print(f"Testing inference with {zip_path}")
    with open(zip_path, 'rb') as f:
        response = requests.post(
            'http://localhost:3001/predict', 
            files={'dicom_zip': f}
        )
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print("Score:", data.get('patient_score'))
        print("Candidates:", len(data.get('top_candidates', [])))
        print("Candidate Views active slices:")
        views = data.get('candidate_views', {})
        for plane, v in views.items():
            print(f"  {plane}: {list(v.keys())}")
    else:
        print("Response text:", response.text)

if __name__ == "__main__":
    test_inference("test_scan.zip")

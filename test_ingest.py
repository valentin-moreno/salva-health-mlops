from src.data.ingest import load_patients

patients = load_patients()

print(patients.head())
print()
print(f"Total de pacientes: {len(patients)}")

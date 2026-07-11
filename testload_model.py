from src.models.load_model import load_model


print("=" * 60)
print("CARGA DEL MODELO")
print("=" * 60)

model = load_model()

print()

print("=" * 60)
print("MODELO")
print("=" * 60)

print(model)

print()

print("✓ Modelo cargado correctamente.")

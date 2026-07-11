from src.data.split import prepare_train_test

from src.models.load_model import load_model
from src.models.predict import predict


print("=" * 60)
print("INFERENCIA DEL MODELO")
print("=" * 60)

_, X_test, _, y_test = prepare_train_test()

model = load_model()

predictions, probabilities = predict(
    model,
    X_test,
)

print()

print("=" * 60)
print("PREDICCIONES")
print("=" * 60)

print(predictions[:10])

print()

print("=" * 60)
print("PROBABILIDADES")
print("=" * 60)

print(probabilities[:10])

print()

print("=" * 60)
print("VALORES REALES")
print("=" * 60)

print(y_test.head(10).values)

print()

print("✓ Inferencia realizada correctamente.")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

print("✅ Librerías importadas correctamente")

# ── BLOQUE 2: MENÚ DEL RESTAURANTE ──────────────────────────

MENU = [
    # (nombre, categoría, precio)
    ("Alitas BBQ",              "entrada",    12.99),
    ("Ceviche de Camarones",    "entrada",    14.99),
    ("Ensalada Caesar",         "entrada",     9.99),
    ("Soup del Día",            "entrada",     7.99),
    ("Filete a la Parrilla",    "principal",  28.99),
    ("Costillas BBQ",           "principal",  24.99),
    ("Burger Premium",          "principal",  16.99),
    ("Pasta Alfredo",           "principal",  15.99),
    ("Pollo a la Plancha",      "principal",  17.99),
    ("Salmon al Limón",         "principal",  26.99),
    ("Limonada Natural",        "bebida",      4.99),
    ("Cerveza Artesanal",       "bebida",      6.99),
    ("Agua Mineral",            "bebida",      2.99),
    ("Tiramisú",                "postre",      8.99),
    ("Brownie con Helado",      "postre",      7.99),
]

NOMBRES_PLATOS = [p[0] for p in MENU]
N_PLATOS = len(MENU)

print(f"✅ Menú cargado: {N_PLATOS} platos")
print(f"   Entradas:    {sum(1 for p in MENU if p[1]=='entrada')}")
print(f"   Principales: {sum(1 for p in MENU if p[1]=='principal')}")
print(f"   Bebidas:     {sum(1 for p in MENU if p[1]=='bebida')}")
print(f"   Postres:     {sum(1 for p in MENU if p[1]=='postre')}")
# ── BLOQUE 3: DATASET DE CLIENTES ───────────────────────────

def generar_clientes(n=2000):
    """
    Genera perfiles sintéticos de clientes de AIA en Florida.
    Cada cliente tiene un perfil y un patrón de pedidos realista.
    """

    # ── Variables del perfil del cliente ──────────────────────
    zonas = ["Miami", "Orlando", "Tampa", "Jacksonville", "Naples"]
    zona        = np.random.choice(zonas, n)
    comensales  = np.random.randint(1, 9, n)
    gasto_prom  = np.round(np.random.lognormal(3.4, 0.5, n).clip(10, 200), 2)
    frecuencia  = np.random.randint(1, 15, n)   # pedidos por mes
    hora_tipica = np.random.randint(8, 23, n)
    es_corporativo = np.random.binomial(1, 0.20, n)  # 20% son empresas

    # ── Codificar zona a número ────────────────────────────────
    le_zona = LabelEncoder()
    zona_cod = le_zona.fit_transform(zona)

    # ── Generar etiquetas: qué platos pide cada cliente ───────
    # Para cada cliente y cada plato, calculamos si lo pide (1) o no (0)
    # basado en su perfil. Esto simula su historial de pedidos.

    etiquetas = np.zeros((n, N_PLATOS))

    for i in range(n):
        g = gasto_prom[i]
        c = comensales[i]
        h = hora_tipica[i]
        corp = es_corporativo[i]

        # Índices de platos por categoría
        # Entradas: 0-3, Principales: 4-9, Bebidas: 10-12, Postres: 13-14

        # ── Entradas ──────────────────────────────────────────
        # Alitas BBQ (0): grupos grandes y gasto medio
        if c >= 3 and g >= 20:
            etiquetas[i, 0] = 1 if np.random.random() < 0.75 else 0

        # Ceviche (1): zona Miami o Naples, gasto alto
        if zona[i] in ["Miami", "Naples"] and g >= 40:
            etiquetas[i, 1] = 1 if np.random.random() < 0.80 else 0

        # Ensalada Caesar (2): clientes con gasto bajo o dieta
        if g < 35 or (h >= 11 and h <= 14):
            etiquetas[i, 2] = 1 if np.random.random() < 0.55 else 0

        # Soup del día (3): hora de almuerzo
        if h >= 11 and h <= 14:
            etiquetas[i, 3] = 1 if np.random.random() < 0.45 else 0

        # ── Principales ───────────────────────────────────────
        # Filete (4): gasto alto
        if g >= 60:
            etiquetas[i, 4] = 1 if np.random.random() < 0.85 else 0

        # Costillas BBQ (5): grupos, gasto medio-alto
        if c >= 2 and g >= 35:
            etiquetas[i, 5] = 1 if np.random.random() < 0.70 else 0

        # Burger (6): cualquier perfil, más probable gasto bajo
        prob_burger = 0.80 if g < 30 else 0.45
        etiquetas[i, 6] = 1 if np.random.random() < prob_burger else 0

        # Pasta Alfredo (7): hora de noche o almuerzo
        if h >= 18 or (h >= 11 and h <= 14):
            etiquetas[i, 7] = 1 if np.random.random() < 0.55 else 0

        # Pollo a la Plancha (8): gasto bajo-medio
        if g < 45:
            etiquetas[i, 8] = 1 if np.random.random() < 0.65 else 0

        # Salmon (9): gasto alto, Miami o Naples
        if g >= 50 and zona[i] in ["Miami", "Naples"]:
            etiquetas[i, 9] = 1 if np.random.random() < 0.75 else 0

        # ── Bebidas ───────────────────────────────────────────
        # Limonada (10): cualquier cliente, muy común
        etiquetas[i, 10] = 1 if np.random.random() < 0.80 else 0

        # Cerveza (11): noche, grupos
        if h >= 17 and c >= 2:
            etiquetas[i, 11] = 1 if np.random.random() < 0.70 else 0

        # Agua (12): corporativo o almuerzo
        if corp or (h >= 11 and h <= 14):
            etiquetas[i, 12] = 1 if np.random.random() < 0.75 else 0

        # ── Postres ───────────────────────────────────────────
        # Tiramisu (13): gasto alto, noche
        if g >= 45 and h >= 18:
            etiquetas[i, 13] = 1 if np.random.random() < 0.65 else 0

        # Brownie (14): grupos con niños o gasto medio
        if c >= 3 and g < 60:
            etiquetas[i, 14] = 1 if np.random.random() < 0.55 else 0

    # ── Armar el DataFrame ────────────────────────────────────
    X = pd.DataFrame({
        "zona"          : zona_cod,
        "comensales"    : comensales,
        "gasto_prom"    : gasto_prom,
        "frecuencia"    : frecuencia,
        "hora_tipica"   : hora_tipica,
        "es_corporativo": es_corporativo,
    })

    y = pd.DataFrame(etiquetas, columns=NOMBRES_PLATOS)

    return X, y, le_zona, zona


X, y, le_zona, zona_raw = generar_clientes(2000)

print(f"✅ Dataset generado: {len(X)} clientes")
print(f"   Variables de entrada : {X.shape[1]}")
print(f"   Platos a predecir    : {y.shape[1]}")
print(f"\n   Popularidad de cada plato:")
for plato in NOMBRES_PLATOS:
    pct = y[plato].mean() * 100
    barra = "█" * int(pct / 5)
    print(f"   {plato:25s} {barra} {pct:.1f}%")

    # ── BLOQUE 4: PREPARAR DATOS ─────────────────────────────────

# Normalizar las variables de entrada a rango [0, 1]
# La red neuronal aprende mejor cuando todos los números
# están en la misma escala
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values,
    test_size=0.20,
    random_state=42
)

# Convertir a tensores de PyTorch
X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train)
y_test_t  = torch.FloatTensor(y_test)

print(f"✅ Datos preparados")
print(f"   Entrenamiento : {X_train_t.shape[0]} clientes")
print(f"   Prueba        : {X_test_t.shape[0]} clientes")
print(f"   Entrada shape : {X_train_t.shape}")
print(f"   Salida shape  : {y_train_t.shape}")

# ── BLOQUE 5: RED NEURONAL ───────────────────────────────────

class RecomendadorAIA(nn.Module):
    """
    Red neuronal para recomendar platos a clientes de AIA.

    Entrada : 6 variables del perfil del cliente
    Salida  : 15 probabilidades independientes (una por plato)

    Cada salida usa Sigmoid — no Softmax — porque un cliente
    puede tener probabilidad alta en varios platos al mismo tiempo.
    """

    def __init__(self):
        super(RecomendadorAIA, self).__init__()

        self.red = nn.Sequential(

            # Capa 1: 6 entradas → 64 neuronas
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa 2: 64 → 128 neuronas
            # Aquí la red EXPANDE antes de comprimir
            # porque necesita aprender combinaciones complejas
            # entre variables (zona + gasto + hora juntos)
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa 3: 128 → 64 neuronas
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Salida: 64 → 15 probabilidades
            nn.Linear(64, N_PLATOS),
            nn.Sigmoid()   # una probabilidad independiente por plato
        )

    def forward(self, x):
        return self.red(x)


modelo = RecomendadorAIA()

total_params = sum(p.numel() for p in modelo.parameters())
print(f"✅ Red neuronal creada")
print(f"   Arquitectura : 6 → 64 → 128 → 64 → 15")
print(f"   Parámetros   : {total_params:,}")
print(f"\n{modelo}")

# ── BLOQUE 6: ENTRENAMIENTO ───────────────────────────────────

# BCELoss: Binary Cross Entropy
# Evalúa cada plato de forma independiente
# Perfecta para clasificación multi-etiqueta con Sigmoid
criterio    = nn.BCELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

N_EPOCAS   = 100
historial  = {"loss_train": [], "loss_val": []}

print(f"\n{'='*50}")
print(f"  ENTRENANDO ({N_EPOCAS} épocas)")
print(f"{'='*50}")

for epoca in range(N_EPOCAS):

    # ── Entrenamiento ──────────────────────────────────────
    modelo.train()
    optimizador.zero_grad()
    pred_train  = modelo(X_train_t)
    loss_train  = criterio(pred_train, y_train_t)
    loss_train.backward()
    optimizador.step()

    # ── Validación ─────────────────────────────────────────
    modelo.eval()
    with torch.no_grad():
        pred_val = modelo(X_test_t)
        loss_val = criterio(pred_val, y_test_t)

    historial["loss_train"].append(loss_train.item())
    historial["loss_val"].append(loss_val.item())

    if (epoca + 1) % 10 == 0:
        print(f"  Época {epoca+1:3d}/{N_EPOCAS} │ "
              f"Loss train: {loss_train.item():.4f} │ "
              f"Loss val: {loss_val.item():.4f}")

print(f"\n✅ Entrenamiento completo")
print(f"   Loss final entrenamiento : {historial['loss_train'][-1]:.4f}")
print(f"   Loss final validación    : {historial['loss_val'][-1]:.4f}")

# ── BLOQUE 7: EVALUACIÓN ─────────────────────────────────────

modelo.eval()
with torch.no_grad():
    probabilidades = modelo(X_test_t).numpy()

# Convertir probabilidades a 0/1 con umbral 0.5
predicciones = (probabilidades >= 0.50).astype(int)
reales       = y_test_t.numpy().astype(int)

# Calcular precisión por plato
print(f"\n{'='*55}")
print(f"  PRECISIÓN POR PLATO")
print(f"{'='*55}")

aciertos_por_plato = []
for i, plato in enumerate(NOMBRES_PLATOS):
    aciertos = (predicciones[:, i] == reales[:, i]).mean() * 100
    aciertos_por_plato.append(aciertos)
    barra = "█" * int(aciertos / 5)
    print(f"  {plato:25s} {barra} {aciertos:.1f}%")

precision_global = np.mean(aciertos_por_plato)
print(f"\n  Precisión global promedio : {precision_global:.1f}%")

# ── Gráfica: curvas de pérdida ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Red Neuronal — Recomendador de Platos AIA", fontsize=13)

axes[0].plot(historial["loss_train"], color="steelblue",
             lw=2, label="Entrenamiento")
axes[0].plot(historial["loss_val"],   color="orangered",
             lw=2, label="Validación", linestyle="--")
axes[0].set_title("Curva de Pérdida")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("BCELoss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Gráfica: precisión por plato ─────────────────────────────
colores = ["steelblue" if a >= 70 else "orangered"
           for a in aciertos_por_plato]
axes[1].barh(NOMBRES_PLATOS, aciertos_por_plato, color=colores)
axes[1].axvline(70, color="gray", linestyle="--", alpha=0.7,
                label="Umbral 70%")
axes[1].set_title("Precisión por Plato")
axes[1].set_xlabel("Precisión (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("outputs/evaluacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Gráfica guardada en outputs/evaluacion.png")

# ── BLOQUE 8: RECOMENDAR A UN CLIENTE ────────────────────────

def recomendar(perfil: dict, top_por_categoria: int = 2):
    """
    Genera una recomendación de orden completa para un cliente.

    Parámetros:
        perfil: diccionario con los datos del cliente
        top_por_categoria: cuántos platos recomendar por categoría
    """
    # Codificar la zona
    zona_num = le_zona.transform([perfil["zona"]])[0]

    # Armar el vector de entrada
    entrada = np.array([[
        zona_num,
        perfil["comensales"],
        perfil["gasto_prom"],
        perfil["frecuencia"],
        perfil["hora"],
        perfil["es_corporativo"],
    ]], dtype=np.float32)

    # Normalizar con el mismo scaler del entrenamiento
    entrada_scaled = scaler.transform(entrada)
    entrada_tensor = torch.FloatTensor(entrada_scaled)

    # Predecir
    modelo.eval()
    with torch.no_grad():
        probs = modelo(entrada_tensor).numpy()[0]

    # Organizar por categoría
    categorias = {
        "🥗 ENTRADAS"  : list(range(0, 4)),
        "🍽️  PRINCIPALES": list(range(4, 10)),
        "🥤 BEBIDAS"   : list(range(10, 13)),
        "🍰 POSTRES"   : list(range(13, 15)),
    }

    # Calcular ticket estimado
    platos_seleccionados = []
    for indices in categorias.values():
        probs_cat = [(i, probs[i]) for i in indices]
        top = sorted(probs_cat, key=lambda x: x[1], reverse=True)[:top_por_categoria]
        platos_seleccionados.extend([MENU[i] for i, p in top if p >= 0.40])

    precios = [p[2] for p in platos_seleccionados]
    ticket_min = round(sum(sorted(precios)[:3]) * perfil["comensales"] * 0.8, 2)
    ticket_max = round(sum(sorted(precios)[-3:]) * perfil["comensales"], 2)

    # Imprimir recomendación
    linea = "─" * 45
    print(f"\n{linea}")
    print(f"  🍽️  RECOMENDACIÓN PARA AIA TECHNOLOGY")
    print(f"  Zona: {perfil['zona']} | "
          f"Comensales: {perfil['comensales']} | "
          f"Gasto prom: ${perfil['gasto_prom']}")
    print(linea)

    for cat, indices in categorias.items():
        probs_cat = [(i, probs[i]) for i in indices]
        top = sorted(probs_cat, key=lambda x: x[1], reverse=True)[:top_por_categoria]
        print(f"\n  {cat}")
        for idx, prob in top:
            nombre = MENU[idx][0]
            precio = MENU[idx][2]
            barra  = "█" * int(prob * 20)
            alerta = " ⭐" if prob >= 0.75 else ""
            print(f"    {nombre:25s} {barra:<20} {prob*100:.0f}%  ${precio}{alerta}")

    print(f"\n{linea}")
    print(f"  💰 Ticket estimado: ${ticket_min} - ${ticket_max}")
    print(linea)


# ── DEMO: 3 tipos de cliente ──────────────────────────────────

print("\n" + "="*45)
print("  DEMO DE RECOMENDACIONES")
print("="*45)

recomendar({
    "zona": "Miami", "comensales": 4,
    "gasto_prom": 85, "frecuencia": 8,
    "hora": 20, "es_corporativo": 0
})

recomendar({
    "zona": "Orlando", "comensales": 1,
    "gasto_prom": 18, "frecuencia": 3,
    "hora": 13, "es_corporativo": 0
})

recomendar({
    "zona": "Tampa", "comensales": 10,
    "gasto_prom": 120, "frecuencia": 12,
    "hora": 19, "es_corporativo": 1
})
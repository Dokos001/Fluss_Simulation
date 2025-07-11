from matplotlib import pyplot as plt
from scipy.stats import invgauss
# Theoretischer Kondensator Wert
epsilon_0 = 8.854e-12  # Permittivität des Vakuums in F/m
epsilon_r1 = 3 # PVC SChlacuh
epsilon_r2 =  80 # Wasser
# Geometrische Parameter des Kondensators
# länge 10 mm
l = 10e-3  # Länge in m (Umrechnung von mm nach m)
# Breite 1.56 mm
w = 1.56e-3  # Breite in m (Umrechnung
 
# Abstand zwischen Schlauch = 1,9 mm
d1 = 0.85e-3  # Abstand in m (Umrechnung von mm nach m)
# Abstand zwischen Wasser und Schlauch = 3,17 mm
d2 = 3.17e-3  # Abstand in m (Umrechnung von mm nach m)
# Fläche und Kapazität berechnen
A = l * w  # Fläche in m² (Umrechnung von mm² nach m²)
print(f"Fläche A: {A:.15f} m²")
C_1 = (epsilon_0 * epsilon_r1 * A) / d1  # Kapazität in F
C_2 = (epsilon_0 * epsilon_r2 * A) / d2  # Kapazität in F
# Gesamte Serienkapazität
C_total = 1 / (2 / C_1 + 1 / C_2)  # Gesamtkapazität in F
print(f"Theoretische Kapazität C1: {C_1:.15f} F")
print(f"Theoretische Kapazität C2: {C_2:.15f} F")
print(f"Theoretische Gesamtkapazität C_total: {C_total:.15f} F")
# print in pF
print(f"Theoretische Gesamtkapazität C_total: {C_total * 1e12:.15f} pF")
# Theoretische Kapazität in pF
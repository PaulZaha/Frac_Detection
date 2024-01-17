import matplotlib.pyplot as plt

# Die gegebenen Daten
data = [1.962491591771444e-05, 1.9068162064964758e-05, 1.839152076817538e-05, 1.7582137830105014e-05, 1.6513348711528517e-05, 
1.5302165247988548e-05, 1.4015883123401402e-05, 1.2767695077019828e-05, 1.1602428708797839e-05, 1.0539783445874007e-05, 
9.487015011218296e-06, 8.497224007344842e-06, 7.585392051963073e-06, 6.823938722878052e-06, 6.14861985227805e-06, 
5.580468720745159e-06]

# Plot erstellen
plt.plot(data, marker='o', linestyle='-')

# Achsentitel hinzufügen
plt.xlabel('Index')
plt.ylabel('Wert')

# Titel hinzufügen
plt.title('Plot der gegebenen Daten')

# Grid hinzufügen
plt.grid(True)

# Plot anzeigen
plt.show()
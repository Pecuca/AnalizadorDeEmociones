import sqlite3

conn = sqlite3.connect("app_data.db")
c = conn.cursor()

# Ejemplo: eliminar persona con id=1
c.execute("DELETE FROM detecciones WHERE persona_id = ?", (1,))
c.execute("DELETE FROM personas WHERE id = ?", (1,))

c.execute("DELETE FROM detecciones WHERE persona_id = ?", (2,))
c.execute("DELETE FROM personas WHERE id = ?", (2,))

c.execute("DELETE FROM detecciones WHERE persona_id = ?", (3,))
c.execute("DELETE FROM personas WHERE id = ?", (3,))

conn.commit()
conn.close()

print("Registros eliminados correctamente.")
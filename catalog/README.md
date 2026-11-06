# Catalog models

Drop licensed 3D models here and the 3D Studio shows them instantly, labelled "Catalog model", whenever the user confirms that car.

- File name is the Stanford Cars class id: `179.glb` is the Spyker C8 Coupe 2009. Ids are in `results/class_names.json`.
- GLB only. The app makes the OBJ and STL downloads from it and keeps the GLB's own materials.
- Only add models you are allowed to use. Keep the licence and author for each one in `catalog/SOURCES.md` (file, car, author, licence, link), because the app does not show them.
- Nothing here is built from the user's photo, so the Studio says so on the page.

Without a file here, the Studio builds the mesh from a reference photo of the matched car (and reuses it next time), or from the user's own photo if the car is not one of the 196 classes.

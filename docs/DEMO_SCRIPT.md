# Apex3D demo script

**Three minutes. Draft:** the backend doesn't exist yet, so rehearse and time this once the real flow works. Items in `[brackets]` must be filled in from real results.

## Before judging

- [ ] Laptop plugged in, sleep off, backend and tunnel running
- [ ] Public link opened on a phone that isn't on the laptop's network
- [ ] Phone hotspot ready, in case the venue Wi-Fi fails
- [ ] No other GPU jobs running (ask Person A)
- [ ] Mesh generated once, so the model is loaded and the timing is known
- [ ] Demo photos on the presenting device, none of them from the training set:
  - A clear photo of a car in the 196 classes, tested beforehand
  - A car that gives an ambiguous result
  - A car newer than 2012, or a van, that is rejected
- [ ] Backup video open in another tab (a recording of the full flow is in `docs/demo/demo.mp4`; screenshots of each screen are in `docs/screenshots/`)

## The script

| Time | Do | Say |
|---|---|---|
| 0:00 | Home page | "Apex3D identifies a car from a photo, shows how sure it is, and helps turn that identity into a draft 3D model. Recognition is the P5 brief, and the 3D is our addition on top." |
| 0:15 | Upload the clear photo | "This is running live on our GPU laptop through a public link." |
| 0:25 | Vehicle Identity. Open the Top-5 | "An R-CNN finds the car, and a ConvNeXt fine-tuned on Stanford Cars classifies the crop. Here's the box, the prediction, the Top-5, and how long each step took." |
| 0:45 | Classical CV page | "The classical pipeline runs on the same photo. These are real intermediate results: CLAHE, edges, contours, ROI, colour and texture features." |
| 1:00 | Upload the newer car or van | "A closed-set classifier always gives an answer. Here it isn't sure, so the app says so and won't let the guess through unless a person picks it." |
| 1:20 | Back to the first car. Confirm, open Reference Studio | "The confirmed identity finds openly licensed photos. Our own classifier checks them, so it works as a filter too. Licences stay attached, and the app is clear these are the same model, not the same car." |
| 1:40 | Generate the mesh, show the queue and logs | "TripoSR makes a mesh from one view. This is the real job status." Reconstruction takes `[time]`, so talk over it or open one generated just before. |
| 2:05 | Rotate, zoom, wireframe. Show validation | "This is the generated file. The validation report says whether it's watertight, and it isn't print-ready. The sides the photo doesn't show are inferred, and the scale is unknown, and the app says that too." |
| 2:25 | Research Lab | "Every number here is read from files our notebooks saved. Top-1 is `[x]`, against `[y]` for classical and `[z]` for hybrid. `[One sentence on whether hybrid helped.]`" |
| 2:45 | Show one failure case | "Here's where it fails: `[case and reason]`." |
| 2:55 | Back to Home | "Recognise, reconstruct, create. Thank you." |

## If something goes wrong

| Problem | Do this |
|---|---|
| Wi-Fi drops | Switch the phone to the hotspot and reload |
| Tunnel URL changed | Use the stable link page, or send the new link |
| Recognition is slow | Keep talking. The elapsed time on screen is real |
| 3D job fails or runs long | Say so honestly, then open a mesh generated earlier and say that it was |
| Laptop fails | Recognition on the second laptop, then the backup video |

Don't present a stored result as live. If it's a saved mesh or a video, say so. The workplan's honesty rule covers the demo as well.

## Likely questions for Person D

- **Why three confidence states?** A percentage alone reads as "probably right". The states change what the user can do.
- **Why no progress bar for 3D?** We have no real percentage, so a bar would be made up. Queue place and logs are real.
- **Where do the Research Lab numbers come from?** JSON files saved by the notebooks. If one is missing, the page says "not available".
- **Why resize photos in the browser?** Phones store rotation as metadata, which could put the box on a rotated image, and smaller files upload faster through the tunnel.
- **What happens if a metrics file has the wrong format?** That one section says so and the rest of the page still works.

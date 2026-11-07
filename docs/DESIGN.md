# Apex3D design document

**Status: current.** This document covers the product and interface decisions, the limits of the system, and the ethical points. The problem, methods and measured results are in the technical report (`report/report_P5_T25.md`, built to PDF), in `eval_P5_T25.md` and in [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md). The system diagram and what is stored where are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Who it's for

The judges come first: they will open a public link on their own phones, probably on unreliable Wi-Fi, with a few minutes to spare. Behind them are 3D artists and students who want a starting mesh and need to know how far they can trust it.

## Design principles

1. **Show uncertainty, don't hide it.** A closed-set classifier always gives an answer, so the interface must show when that answer is weak.
2. **The person decides.** Each stage produces something to inspect and correct before the next one runs.
3. **Only show what is real.** No hardcoded metrics, simulated progress or stored results shown as live.
4. **A failure is a normal state.** Every request has a designed loading and error state, and most errors offer a way forward.

## Key decisions

| Decision | Why | Alternative rejected |
|---|---|---|
| Three visible confidence states, each changing the page | Confident results need one tap. Ambiguous and rejected results require the user to choose, so a weak guess can't pass through unnoticed | A percentage alone, which people read as "probably right" even when it's low |
| The confirm button stays disabled until a class is picked in ambiguous and rejected states | Forces a decision at the point where the model is unsure | Pre-selecting the top guess |
| "None of these" is always available | The true class may not be among the 196, and the user can still make a mesh | Making the flow depend on a confirmed class |
| Manufacturer, Model and Year are separate labelled fields on the identity page | The task asks for all three, and a single label like "Audi R8 Coupe 2012" hides which part is wrong | One combined label |
| Everything sits behind a login | The server runs models and stores photos, so only people with an account should reach it. Passwords are stored as scrypt hashes and the session is an HttpOnly cookie | Open access, or a login on only some pages |
| The 3D Studio shows a designed showcase car first, with the TripoSR mesh as a second tab | Single-photo reconstruction gives lumpy, unreliable shapes. The showcase car always has a clean structure, takes its paint colour from the photo and its proportions from the body style, and downloads in three formats. The mesh stays available and labelled experimental | Presenting the TripoSR mesh as the main result |
| The job page opens on the experimental mesh, and the pre-build page has the same tabs | Someone who just asked for a mesh should see its progress, and the two pages should not jump around | Opening on the showcase after a build request |
| Photos for recognition are rotated and resized in the browser before upload | Phones store rotation as metadata that servers often ignore, which would put the box on a rotated image. Smaller files also upload faster | Relying on each server step to handle rotation |
| Job status comes only from the server, with an indeterminate spinner | There is no real percentage to show, so a progress bar would be invented. Queue place and logs are real | A timed progress bar |
| The job id is in the 3D Studio address | Lets a phone survive a reload or a dropped connection, and the result can be reopened from a link | Holding the job only in memory |
| Polling keeps retrying through network drops | The job carries on at the server, so a Wi-Fi blip shouldn't look like a failure | Treating any failed poll as a failed job |
| Uploaded photos are deleted automatically after 2 hours | Photos can show people, plates and places, so they should not be kept. The cost is that a result link stops working after the retention period, and the page says so | Keeping uploads until someone cleans up by hand (this was the behaviour before, and it kept photos indefinitely) |
| The live recognition path is detector, crop, classifier. Classical features are shown in their own view and used as the baseline | Fusing classical geometry features with the deep embedding was tested and did not improve accuracy, so it was left out of the production path | Claiming a hybrid gain that the numbers do not show |
| The Home hero is a real 3D car, with its grid drawn in a shader | The grid is computed from each surface point's position, so it follows the real shape instead of sitting on top as a flat image. It also loads as a separate download, so the headline and buttons never wait for it | A looping video or a rendered image |
| 3D code loads only when needed | three.js is about 70% of the JavaScript, and it would slow the core P5 flow on phones | One bundle |
| Canvas draws on demand | Saves phone battery and heat while the mesh is still | Drawing continuously |
| Fonts are bundled, with no external requests | The only server the app needs is its own, which also suits the privacy statement | Font CDN |
| Research Lab sections fail individually | Each file comes from a different notebook and may arrive late or change format. The rest of the page must still work | One error for the whole page |

## Honesty in the interface

| Limit | Where the user sees it |
|---|---|
| Only 196 classes, up to 2012 | Home and Vehicle Identity |
| No vehicle detected | Warning beside the photo on Vehicle Identity |
| References show the same model, not the same car | Reference Studio |
| Unconfirmed references | "Approximate reference" label on the image |
| Attribution and licence | Beside each reference, in Reference Studio and 3D Studio, and under the showcase car |
| Unseen sides are inferred, scale unknown | 3D Studio, before and after generation |
| Not print-ready | Under the validation report, which never gives a pass verdict |
| Missing results | "Not available" in the Research Lab |
| Fusion did not help | The hybrid row in the Research Lab's comparison table |
| Photos are processed on the server, and a copy sent for a 3D build is kept for up to 2 hours | Upload page |

## Limitations

Every figure below is measured in `results/` and reported in `eval_P5_T25.md`.

**What the model cannot do**

- **It only knows 196 cars, none newer than 2012.** It always answers with one of them. On photos outside the dataset, the confidence rule rejected 91.5% of other vehicles (airplanes, motorbikes, helicopters, boats) and 98.3% of other things, but still gave a confident wrong answer for 8.5% and 1.7% of them. That test used Caltech-101 photos, which contain no newer cars, so it does not measure the most likely real case: a modern car the model has never seen.
- **It is wrong about 1 in 13 times.** Top-1 on the 8,041 test images is about 92%. Of the errors, 63.5% are the same make and 6.2% are the same make and model with a different year, so mistakes are mostly between near-identical cars. Make-level accuracy is 97.2%.
- **"Confident" is not "certain".** Images in the confident state were right 95.0% of the time. Of the 642 errors, 196 had a confidence of at least 90%. One of the most confident errors looks like a wrong dataset label rather than a model mistake, so the true accuracy may be slightly higher than reported, but we could not check labels independently.
- **Weak conditions.** Accuracy falls to 52.7% under heavy blur, 51.5% at 64 px, 9.0% at 32 px and 60.7% with 40% of the image hidden. Adaptive CLAHE did not help in low light (79.0% without it, 76.8% with it, at the darkest level).

**What the evaluation does not show**

- The headline accuracy uses the dataset's annotated boxes. With the detector's box it was 92.4% on a random sample of 500 test images (the margin of error is about 2.3 points), so it is a sample, not the full set.
- The robustness test used 1,000 sampled test images and synthetic corruptions, not real bad photos.
- Detector mAP is understated, because the dataset marks one car per image and correct detections of other cars count as mistakes.
- Timings come from two different laptops and a CPU, not from the phones judges will use.
- The classical baseline is weak (17.3% Top-1), and fusing its features with the deep model gave no gain (92.18% deep only, 92.09% fused). We report that as it is.

**The 3D part**

- TripoSR builds a mesh from one photo, so the unseen sides are guessed, the scale is unknown, and the shapes can be lumpy. A mesh took about 4 to 5 minutes on CPU when timed here. In the example we validated, the mesh was watertight but flagged "Needs repair".
- The showcase car is one generic model re-coloured and re-proportioned by body style. It is not a copy of the identified car, and the page says so.
- Meshes are not print-ready and the validation report never says they are.

**The system**

- One server, one mesh built at a time, and the rate limits on logins are kept in memory, so they reset when the server restarts.
- There is no password reset and no email verification, so anyone can sign up with any email address.
- It has not been load-tested, and it has not been tested on a real phone, on iOS Safari (including HEIC photos), or on a public URL. **There is no public URL yet.** Deployment files exist for Render and Hugging Face, but neither free option can run the whole app (see `docs/DEPLOY_RENDER.md` and `docs/DEPLOY_HUGGINGFACE.md`).

## Ethics

**Privacy**

- Photos can show people, number plates and places. The app never reads plates and has no code that tries to. It identifies make, model and year only.
- A photo sent for recognition is held in memory and is not written to disk. A photo sent for a 3D build, or uploaded as a reference, is written to disk and deleted automatically after 2 hours (`APEX3D_RETENTION_HOURS`), checked at start-up and every 15 minutes. Before this was added, the server kept such photos until someone ran a cleanup by hand, and 38 photos had piled up on the development laptop.
- The server asks Wikimedia Commons for reference photos by class name. User photos are never sent to Wikimedia or any other outside service.
- The server stores each account's name, email and a salted scrypt hash of the password. Sessions are stored as hashes of a random token. There are no analytics or trackers, and fonts are bundled.
- Server logs can contain request paths and file names. They are not shipped anywhere.

**Consent and access**

- Signing up takes a name, an email and a password. Nothing verifies the email, so the accounts prove nothing about who someone is.
- Sign-ups are limited to 30 an hour and wrong passwords to 5 per email per 10 minutes, to slow abuse. That limit can also be used to lock out a known email address for 10 minutes.

**Licences and credit**

- Reference photos come from Wikimedia Commons with their licence and author shown. The showcase car is the Khronos "Car Concept" model (CC BY 4.0), credited in the app. TripoSR is MIT-licensed, and the background remover is pinned to u2net (Apache-2.0), because the newer default is non-commercial.
- The dataset is Stanford Cars, which is for research use. The repository includes a small number of its test photos as failure examples (six classifier cases and five detector or classical cases, as thumbnails). If the dataset's terms require it, those should be removed before the repository is made public.

**Misuse and fairness**

- A tool that names a vehicle from a photo could be used to follow a particular car. Apex3D does not link a photo to an owner, keep a history, or search across users, and it is not meant for tracking or law enforcement.
- The model is least reliable on near-identical models and on cars outside its 1990 to 2012 range, so a wrong answer is more likely for those. Per-class accuracy is in `results/classifier_per_class.csv`. We did not analyse accuracy by region of the world or by manufacturer group, so we cannot say the model is equally good for all of them.

**Using AI**

- AI assistance (Claude, through Claude Code) was used to build much of this repository: the frontend, the backend integration and login, the evaluation scripts, the deployment files and these documents. The report declares this and states the team's own part.

## Visual style

Dark and minimal, as the brief asks. Near-black surfaces, with one orange accent saved for the main action on each page and the detection box.

- **Type:** Instrument Sans for reading, JetBrains Mono for measured values, so numbers look like readouts.
- **Status colours:** green, amber and red are used only for meaning, and always with an icon and a word, so colour-blind users get the same information.
- **Charts:** one blue for a single series and blue plus orange for two, checked with a colour-vision validator against the chart background. Each chart has a table view. Robustness uses small multiples, because six lines on one chart would be hard to separate.
- **Motion:** the detection box, the home-page curve and the hero car animate. All respect the reduced-motion setting, and the hero car pauses when scrolled out of view.

## Phone behaviour

- Bottom navigation in thumb reach, with touch targets of at least about 40 px.
- A "Take a photo" button appears on touch devices only.
- Layouts have no sideways scrolling at 375 px wide. Wide tables scroll inside their own box.
- In the viewer, one finger rotates, two fingers pan and pinching zooms, with instructions that match the device.
- **Known trade-off:** dragging on the viewer rotates the mesh rather than scrolling the page. Users scroll using the area around it.

## Testing so far

- **Automated:** the frontend has 71 tests and the backend 130, covering the API, the login and its protections, the model wrapper, the evaluation tools, the retention task and the report tools. Seven further tests need a teammate's local detector files and were not run here.
- **By hand:** the full flow (sign up, upload, identity, references, 3D Studio, Research Lab) was driven in a desktop browser against the real model and the real API, including a real TripoSR mesh build, and the failure states (bad files, server errors, a failed job, malformed metrics files, no vehicle found) were injected on the earlier test responses.
- **Not yet tested:** a real phone, iOS Safari and HEIC photos, many users at once, and any public URL or hosting platform (the Docker image has not been built).

## Still to do

- [ ] A public URL (the rubric requires one)
- [ ] Run the quickstart on a clean clone and fix anything it finds
- [ ] Test on real phones and send the bug list to the team
- [ ] Record the backup demo video (the screen recording is in `docs/demo/demo.mp4`; check it still matches the app)

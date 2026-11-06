# Apex3D design document

**Status: draft.** The organisers haven't supplied the required template, so these sections may need rearranging once it arrives. This draft covers the interface. For the problem, dataset, models and evaluation plan, see [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md). For the system diagram, see [ARCHITECTURE.md](ARCHITECTURE.md).

No results appear here because nothing has been measured yet.

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
| Photos are rotated and resized in the browser before upload | Phones store rotation as metadata that servers often ignore, which would put the box on a rotated image. Smaller files also upload faster over the tunnel | Relying on each server step to handle rotation |
| Job status comes only from the server, with an indeterminate spinner | There is no real percentage to show, so a progress bar would be invented. Queue place and logs are real | A timed progress bar |
| The job id is in the 3D Studio address | Lets a phone survive a reload or a dropped connection, and the result can be reopened from a link | Holding the job only in memory |
| Polling keeps retrying through network drops | The job carries on at the server, so a Wi-Fi blip shouldn't look like a failure | Treating any failed poll as a failed job |
| Photos are held in memory only | The proposal promises uploads aren't retained. The cost is that a reload loses the photo, and the page says so | Saving to browser storage |
| The Home hero is a real 3D car, with its grid drawn in a shader | The grid is computed from each surface point's position, so it follows the real shape instead of sitting on top as a flat image. It also loads as a separate download, so the headline and buttons never wait for it | A looping video or a rendered image |
| 3D code loads only when needed | three.js is about 70% of the JavaScript, and it would slow the core P5 flow on phones | One bundle |
| Canvas draws on demand | Saves phone battery and heat while the mesh is still | Drawing continuously |
| Fonts are bundled, with no external requests | The only server the app needs is the laptop, which also suits the privacy statement | Font CDN |
| Research Lab sections fail individually | Each file comes from a different notebook and may arrive late or change format. The rest of the page must still work | One error for the whole page |

## Honesty in the interface

| Limit | Where the user sees it |
|---|---|
| Only 196 classes, up to 2012 | Home and Vehicle Identity |
| No vehicle detected | Warning beside the photo on Vehicle Identity |
| References show the same model, not the same car | Reference Studio |
| Unconfirmed references | "Approximate reference" label on the image |
| Attribution and licence | Beside each reference, in Reference Studio and 3D Studio |
| Unseen sides are inferred, scale unknown | 3D Studio, before and after generation |
| Not print-ready | Under the validation report, which never gives a pass verdict |
| Missing results | "Not available" in the Research Lab |
| Uploads go to a team laptop | Home and Upload |

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

Tested in an emulated desktop browser and a 375 × 812 phone viewport, against **test responses only**. Failure states were tested by injecting them: bad files, server errors, a failed job, malformed metrics files, no vehicle found.

**Not yet tested:** a real phone, the real API, the tunnel, real meshes from TripoSR, and iOS Safari, including HEIC photos.

## Still to do

- [ ] Replace this draft's structure with the organisers' template when it arrives
- [ ] Connect to the real API and adjust `frontend/src/api/` to Person C's formats
- [ ] Test real TripoSR meshes in the viewer, including their orientation
- [ ] Test on real phones through the tunnel and send the bug list to C and A
- [ ] Record the backup demo video

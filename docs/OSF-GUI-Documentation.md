# APGI Open Science Framework — User Guide

**For:** Researchers managing APGI Empirical Protocols (EP-0 through EP-6)  
**Tool:** `OSF_GUI.py` — Protocol tracking, pre-registration, and report export  
**Version:** 1.0.0

---

## What This Tool Does

The OSF GUI is your central dashboard for managing the seven empirical protocols that make up the APGI validation programme. It lets you:

- **Track progress** — see at a glance which protocols are pre-registered, in data collection, or completed
- **Generate pre-registration templates** — produce a structured OSF-ready document for any protocol
- **Export status reports** — create a full summary of all seven protocols (as plain text or JSON)
- **Save and restore your work** — snapshot your current status entries and reload them later

It does not connect to OSF automatically. Templates and reports are saved to your local machine and uploaded to OSF manually.

---

## Starting the App

Open a terminal, navigate to the project folder, and run:

```bash
python OSF_GUI.py
```

The window opens at 1020 × 740 px. You can resize it — the layout adapts.

On first launch, all protocols show **Not started / Pending**. Your status entries are kept in memory until you save a snapshot (see [Saving Your Progress](#saving-and-restoring-progress)).

---

## The Interface at a Glance

```text
┌──────────────────────────────────────────────────────────────────┐
│  APGI OPEN SCIENCE FRAMEWORK — EMPIRICAL PROTOCOLS               │
│  Protocols: 7  |  Pre-registered: 0/7  |  Completed: 0/7  OSF: Ready │
├──────────────────┬───────────────────────────────────────────────┤
│ EMPIRICAL        │  [Protocol cards — type, deps, status]        │
│ PROTOCOLS        │                                               │
│                  │  [Detail panel — description, measures,       │
│  EP-0  HEP …     │   analysis plan, falsification criterion,     │
│  EP-1  EEG …     │   status dropdowns, notes, Save button]       │
│  EP-2  TMS …     │                                               │
│  EP-3  Active … │                                               │
│  EP-4  DOC …     │                                               │
│  EP-5  fMRI …    │                                               │
│  EP-6  iEEG …    │                                               │
│                  │                                               │
│  ✎ Generate Pre-reg                                              │
│  Export Report                                                   │
│  Dependency Overview                                             │
│  Clear Console                                                   │
├──────────────────┴───────────────────────────────────────────────┤
│  ▼ OUTPUT CONSOLE                                                │
│  [14:32:15]  OSF Protocol Manager initialised. Select a protocol.│
└──────────────────────────────────────────────────────────────────┘
```

**Top bar** — live counter showing how many protocols are pre-registered and completed.  
**Sidebar** — click any EP to load it in the workspace.  
**Workspace** — summary cards on top, full editable detail below.  
**Console** — timestamped log of every action you take. Click the ▼ header to collapse it.

---

## The Seven Empirical Protocols

| ID   | Title                              | Type                | Priority   | Requires   |
|------|------------------------------------|---------------------|------------|------------|
| EP-0 | HEP Proxy Validation                | Empirical           | High       | —          |
| EP-1 | EEG Interoceptive Precision Gating  | Empirical           | High       | EP-0       |
| EP-2 | Causal TMS Insula/dlPFC             | Empirical           | Medium     | EP-0       |
| EP-3 | Active Inference Simulations        | Computational       | High       | —          |
| EP-4 | Disorders of Consciousness          | Clinical            | Long-term  | EP-0       |
| EP-5 | fMRI Anticipation vs. Experience     | Empirical           | High       | EP-0       |
| EP-6 | iEEG All-or-None Dynamics           | Clinical-Empirical  | Medium     | EP-3       |

**Dependency rule:** EP-0 must be completed before EP-1, EP-2, EP-4, or EP-5 can begin. EP-3 must be completed before EP-6. EP-0 and EP-3 have no upstream requirements.

To see the full dependency chain at any time, click **Dependency Overview** in the sidebar.

---

## Core Workflows

### 1. Reviewing a Protocol

1. Click any protocol name in the sidebar (e.g. **EP-1  EEG Interoceptive Precision Gating**).
2. Four summary cards appear at the top of the workspace:
   - **Protocol** — ID, study type, priority level
   - **Dependencies** — which earlier protocol this requires
   - **Study Status** — your current progress stage
   - **Pre-registration** — pre-reg status and platform (OSF)
3. Scroll down in the detail panel to read the full description, sample size requirement, key measures, analysis plan, and the pre-specified **falsification criterion**.

Nothing changes until you explicitly save — reviewing is always read-only.

---

### 2. Updating Protocol Status

After selecting a protocol:

1. Scroll to the **PROTOCOL DETAILS & STATUS** section at the bottom of the workspace.
2. Use the **Study Status** dropdown to record where the protocol currently stands:

   | Status          | When to use                                                     |
   |-----------------|----------------------------------------------------------------|
   | Not started     | No work has begun                                              |
   | In preparation  | Materials, ethics, or infrastructure being prepared              |
   | Pre-registered  | Study plan submitted and locked on OSF                          |
   | Data collection | Actively recruiting or collecting data                         |
   | Analysis        | Data collected; statistical analysis underway                   |
   | Completed       | Final results documented                                       |
   | On hold         | Paused (note the reason in the Notes field)                    |

3. Use the **Pre-reg Status** dropdown to track registration specifically:

   | Status           | When to use                                                     |
   |------------------|----------------------------------------------------------------|
   | Pending          | No pre-registration action taken yet                           |
   | Draft ready      | Template complete; not yet submitted                          |
   | Submitted        | Submitted to OSF, awaiting confirmation                       |
   | Registered       | Locked and timestamped on OSF                                  |
   | Not applicable   | Protocol type does not require pre-registration              |

4. Add any context in the **Notes** box (e.g. ethics board reference, recruitment site, deviations from plan).
5. Click **Save Changes**. The top counter updates immediately and a confirmation appears.

> **Tip:** Changes are held in memory. Use **File → Save Status Snapshot** (or `Cmd+S`) after each session so your entries survive the next launch.

---

### 3. Generating a Pre-registration Template

When a protocol is ready to pre-register on OSF, the app generates a structured document you can paste directly into OSF's pre-registration form.

1. Select the protocol in the sidebar.
2. Click **✎ Generate Pre-reg** in the sidebar (or **File → Generate Pre-reg Template**, or `Cmd+G`).
3. A preview window opens showing the completed template with six sections:
   - Research Question & Hypotheses
   - Experimental Design (sample size, key measures)
   - Analysis Plan
   - Falsification Criteria
   - Dependencies
   - Notes
4. Review the content, then click **Save to File…** to write it as a `.txt` file.
5. Upload that file to your OSF project and lock the registration.
6. Come back and update **Pre-reg Status → Registered** (see step 2 above).

> The falsification criterion section is pre-filled from the protocol definition and should not be altered after registration — this is your primary commitment to the OSF reviewer.

---

### 4. Exporting a Full Protocol Report

To share a snapshot of all seven protocols with collaborators or attach it to a grant application:

1. Click **Export Report** in the sidebar, or use **File → Export Protocol Report** (`Cmd+E`).
2. A file save dialog opens. Choose a format:
   - **`.txt`** — plain-text report, one section per protocol. Human-readable.
   - **`.json`** — machine-readable. Useful if you want to import the data into another tool or script a summary.
3. The report includes, for every protocol: ID, title, type, priority, dependencies, study status, pre-reg status, and notes.

---

### 5. Saving and Restoring Progress

Your status entries, pre-reg statuses, and notes are **not** saved automatically. Use snapshots to persist your work across sessions.

**Save a snapshot:**

- **File → Save Status Snapshot** (or `Cmd+S`)
- Choose a filename (default: `osf_status_snapshot.json`) and location.
- Recommended: keep a dated copy alongside your OSF project files (e.g. `osf_snapshot_2026-06-03.json`).

**Restore from a snapshot:**

- **File → Load Status Snapshot**
- Select the `.json` file you saved previously.
- All status fields and notes are restored. The workspace refreshes immediately if a protocol is selected.

> If the file is invalid or fields are missing, you will see an error message and no data will be changed. Your current session is preserved.

---

### 6. Viewing Protocol Dependencies

To understand the required execution order before planning data collection:

1. Click **Dependency Overview** in the sidebar (or **View → Dependency Overview**).
2. A scrollable window lists all seven protocols with their type, priority, dependency chain, and current status.
3. This view is read-only — it is intended as a quick reference before scheduling work.

**Key rules from the dependency graph:**

- EP-0 is the gateway study. Its HEP baseline signatures are required by EP-1, EP-2, EP-4, and EP-5.
- EP-3 (computational simulations) is independent and can run in parallel with EP-0.
- EP-6 requires EP-3 to be complete.

---

## Menus and Keyboard Shortcuts

### File Menu

| Action                     | Shortcut   | What it does                                        |
|----------------------------|------------|-----------------------------------------------------|
| Export Protocol Report     | `Cmd+E`    | Save full report (TXT or JSON)                     |
| Generate Pre-reg Template  | `Cmd+G`    | Generate pre-reg doc for selected protocol          |
| Save Status Snapshot       | `Cmd+S`    | Save all statuses and notes to a JSON file         |
| Load Status Snapshot       | —          | Restore statuses from a saved JSON file            |
| Quit                       | `Cmd+Q`    | Close the application                              |

### View Menu

| Action                | Shortcut   | What it does                                           |
|-----------------------|------------|-------------------------------------------------------|
| Clear Console         | `Cmd+L`    | Erase all entries from the output console             |
| Toggle Console        | —          | Collapse or expand the console panel                   |
| Dependency Overview   | —          | Open the dependency graph window                      |

> On Windows or Linux, replace `Cmd` with `Ctrl`.

---

## Understanding the Console

The output console (bottom panel) logs every action with a timestamp:

```
[14:32:15]  OSF Protocol Manager initialised. Select a protocol to begin.
[14:32:15]  7 protocols loaded  |  All status: Not started  |  Pre-reg: Pending
[14:32:45]  Selected: EP-0 — HEP Proxy Validation
[14:33:02]  [EP-0] Status saved: In preparation | Pre-reg: Draft ready
[14:33:15]  [EP-0] Pre-registration template generated.
[14:41:00]  Status snapshot saved: /Users/you/osf_status_snapshot.json
```

Use it to verify that saves and exports completed successfully, especially before closing the app. Click **▼ OUTPUT CONSOLE** to collapse it if you need more workspace height.

---

## Falsification Criteria — What They Mean

Each protocol carries a pre-specified falsification criterion. These are not optional thresholds — they define, in advance, what result would count as evidence against the APGI model:

| Protocol | Falsification criterion                                                                    |
|----------|-------------------------------------------------------------------------------------------|
| EP-0     | HEP modulation < 0.1 μV between high/low interoceptive precision conditions               |
| EP-1     | No GFP modulation by interoceptive precision prior (Bayes factor B < 1/3)                  |
| EP-2     | Insula TMS fails to modulate precision-weighted prediction errors (p > 0.05, d < 0.3)   |
| EP-3     | Active inference agent fails to minimise free energy (>5% residual error at convergence) |
| EP-4     | HEP signatures fail to discriminate VS/UWS from MCS (AUC ≤ 0.70)                        |
| EP-5     | No vmPFC/insula dissociation between anticipatory and reactive conditions (p > 0.05 FWE)  |
| EP-6     | Graded (rather than threshold/all-or-none) responses in insular/ACC circuits              |

These criteria are shown in the detail panel for each protocol. They appear verbatim in the pre-registration template — treat them as binding commitments once registered.

---

## Frequently Asked Questions

**Does this app upload anything to OSF automatically?**  
No. Everything is local. Templates and reports are files you save and then upload to OSF manually.

**Can I edit the protocol descriptions or falsification criteria?**  
Not through the GUI. The protocol definitions are fixed in the source. Only study status, pre-reg status, and notes are editable at runtime.

**What happens if I close without saving?**  
Any status changes made since the last snapshot are lost. Always use **File → Save Status Snapshot** before closing.

**Can I run multiple protocols in parallel?**  
Yes — EP-0 and EP-3 can start simultaneously since neither depends on the other. All other protocols must wait for their upstream requirement to complete.

**The app opened but shows no protocols — what happened?**  
This should not occur. If the sidebar is empty, close and reopen. If the issue persists, run `python OSF_GUI.py --headless` in the terminal — a successful startup should print all seven protocol names.

**What is headless mode?**  
Running `python OSF_GUI.py --headless` starts the app without a visible window and immediately exits after confirming all protocols loaded correctly. It is used to verify the installation is working — not for routine research use.

---

## Typical Session Workflow

A typical session might look like this:

1. **Launch** `python OSF_GUI.py`
2. **Load your snapshot** — File → Load Status Snapshot → select last session's file
3. **Review progress** — click each EP; check the top counter bar
4. **Update any changed statuses** — select protocol → change dropdown → add note → Save Changes
5. **Generate a pre-reg template** if a protocol is ready for OSF submission
6. **Export a report** if needed for a collaborator or grant update
7. **Save snapshot** before closing — File → Save Status Snapshot

---

## Troubleshooting

| Problem                                    | Likely cause              | What to do                                                  |
|-------------------------------------------|---------------------------|-------------------------------------------------------------|
| Status changes not persisting after restart | Snapshot not saved        | Use **File → Save Status Snapshot** before closing           |
| Pre-registration template button is greyed out | No protocol selected   | Click a protocol name in the sidebar first                    |
| Console panel takes up too much space       | —                         | Click **▼ OUTPUT CONSOLE** to collapse it                     |
| Fonts look incorrect                        | Noto Sans not installed   | Install the Noto Sans font family for your OS                |
| Report saved as JSON but looks unreadable   | JSON is machine format    | Use `.txt` extension for a human-readable report             |
| Load snapshot shows an error                | File is corrupt or wrong format | Use only `.json` files saved by this app                   |

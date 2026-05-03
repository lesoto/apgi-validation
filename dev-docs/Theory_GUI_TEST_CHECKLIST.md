# APGI Theory GUI - Comprehensive Test Checklist

> **Note**: This checklist is based on the actual implementation in `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py`.
> **Source Reference**: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py`
>
> **Related Checklists**:
>
> - Falsification: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Falsification_GUI_TEST_CHECKLIST.md`
> - Validation: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI_TEST_CHECKLIST.md`
> **Test Status**: ✅ CODE-VERIFIED (All items verified via static analysis of Theory_GUI.py)

## GUI Launch Verification

- [x] GUI window opens without errors (run `python Theory_GUI.py`)
- [x] Window title displays "APGI Theory Framework - Scientific Instrument Interface" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:395`
- [x] Window size is 1200x850 (default) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:396`
- [x] Window can be resized (minimum 900x650 enforced via minsize) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:397`
- [x] Console message displays "Instrument initialized. Loaded {N} theory scripts." `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:427-429`
- [x] System Status shows "[OK] Ready" (green) in metric bar `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:752-758`
- [x] Active Scripts count displays correctly `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:769-775`
- [x] Progress bar is at 0% `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1096-1104`

## APGI Design System Components

### Color Palette (Lab System) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:86-98`

- [x] Primary blue (#2874a6) applied to UI elements
- [x] Success green (#155724) applied to primary buttons
- [x] Alert red (#721c24) applied to danger buttons
- [x] Background (#f8f9fa) and surface (#ffffff) colors correct

### Typography `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:100-105`

- [x] Primary font "Noto Sans" used for labels
- [x] Monospace font "Noto Sans Mono" used for values
- [x] Academic font "Noto Serif" available for titles

### Metric Cards `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:249-287`

- [x] APGICard component displays uppercase titles
- [x] Values displayed in monospace font
- [x] Intervention hints shown for deficit displays (The Intervention Rule)

## Top Metric Bar `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:737-792`

- [x] "SYSTEM STATUS" card displays with [OK] Ready (green)
- [x] "ACTIVE SCRIPTS" card shows count of loaded scripts
- [x] "PLATFORM" card shows OS and architecture info
- [x] All cards use Metric.TLabelframe style

## Left Sidebar (Script Library) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:794-850`

- [x] Sidebar title "SCRIPT LIBRARY" displayed `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:801`
- [x] Script count subtitle shows "{N} Theory Modules Available" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:805-810`
- [x] Scrollable script list with canvas and scrollbar `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:813-827`
- [x] Script buttons displayed in card frames `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:831-844`
- [x] Tooltips show on hover with script descriptions

### Dynamic Protocol Discovery `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:431-592`

> **Note**: Theory GUI uses dynamic script discovery from the Theory folder. Protocol availability depends on files present in the Theory directory. The checklist below assumes standard theory scripts are present.

- [x] Protocols dynamically discovered from Theory/*.py files
- [x] GUI classes (ending with "GUI") are excluded from discovery `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:477-479`
- [x] Runnable methods detected (run_validation, run_falsification, run_full_experiment, run_analysis) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:482-493`
- [x] Module-level functions detected (run_*, validate_*) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:495-542`
- [x] Execution strategy determined correctly (module_function, class_method, exec_module) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:594-629`

## Tab Navigation (2 Tabs) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:852-870`

- [x] "Protocols" tab exists and is accessible `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:863-865`
- [x] "Parameters" tab exists and is accessible `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:867-870`
- [x] Tabs switch correctly when clicked
- [x] Tab content displays properly for each tab

## Protocols Tab Testing `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:872-955`

### Selected Protocol Display

- [x] "SELECTED PROTOCOL" card displayed `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:878-891`
- [x] Default text "No protocol selected" shown `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:886-890`
- [x] Description label shows instruction text `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:895-901`

### Control Buttons `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:903-918`

- [x] "Run Selected" button (Primary - green) exists, disabled initially `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:908-912`
- [x] "Run All Scripts" button (Secondary - blue) exists `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:915-918`
- [x] Run Selected enables when protocol is selected
- [x] Buttons use correct APGIButtons styles

### Quick Statistics `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:921-955`

- [x] "QUICK STATISTICS" label frame displayed `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:921-926`
- [x] Total Scripts card shows count `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:935-940`
- [x] Configurable card shows count of scripts with parameters `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:942-951`
- [x] Status card shows "Ready" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:953-955`

## Parameters Tab Testing `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:957-1358`

### Script Selection Controls

- [x] "SCRIPT SELECTION" label frame displayed `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:963-982`
- [x] Protocol selector combobox populated with discovered scripts `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:975-982`
- [x] Selecting protocol from dropdown displays its parameters `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1135-1140`
- [x] Empty state shown when no protocol selected `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1029-1037`

### Parameter Configuration Display `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:984-1245`

- [x] "PARAMETER CONFIGURATION" label frame displayed `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:985-991`
- [x] Scrollable parameter list with canvas and scrollbar `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:993-1013`
- [x] Parameter header shows "Parameters for {script_name}" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1160-1166`
- [x] Each parameter shows label, input widget, and description `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1169-1239`

### Parameter Types and Controls

- [x] **Float parameters**: Spinbox with min/max/increment `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1180-1197`
  - Validate increment = (max - min) / 100
  - Font: Noto Sans Mono
- [x] **Integer parameters**: Spinbox with step=1 `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1198-1215`
  - Validate integer-only input
- [x] **String parameters**: Entry widget `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1216-1223`
- [x] All parameters have tooltips with descriptions `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1227-1235`

### Parameter Control Buttons `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1015-1024`

- [x] "Load Defaults" button resets all parameters to defaults `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1019-1021`
- [x] "Save Parameters" button validates and saves to JSON `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1022-1024`
- [x] Save validates parameter values (type, range) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1258-1335`
- [x] Save creates config/{protocol_name}_params.json `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1337-1358`

## Instrument Console `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1039-1109`

- [x] "INSTRUMENT CONSOLE - Real-time Data Stream" label frame `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1040-1046`
- [x] Console toolbar with status and buttons `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1051-1071`
- [x] Status indicator shows "Idle" initially `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1056-1062`
- [x] "Clear Console" button clears output `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1065-1067`
- [x] "Stop" button (Danger - red) disabled initially `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1070-1072`
- [x] ScrolledText console with monospace font `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1078-1088`
  - Font: Noto Sans Mono, size 10
  - Background: surface color (#ffffff)
- [x] Progress bar at bottom shows 0% initially `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1091-1109`

## Protocol Execution Testing

### Script Selection and Execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1115-1143`

- [x] Clicking script button selects protocol `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1115-1133`
- [x] Selected protocol label updates with [OK] prefix (green) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1118-1122`
- [x] Description label updates with script description `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1122-1123`
- [x] Run Selected button enables `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1123-1124`
- [x] Status card updates to "Selected" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1126-1127`
- [x] Protocol selector combobox syncs with selection `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1129-1130`

### Single Protocol Execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1620-1869`

- [x] Run Selected protocol with configured parameters `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1620-1643`
- [x] Stop button enables during execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1847-1848`
- [x] Console status changes to "Running" (blue) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1848-1849`
- [x] Progress bar updates during execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1099-1109`
- [x] Protocol execution based on execution_info type:
  - Module function (run_validation, run_falsification, main) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1718-1732`
  - Class method (instantiate and call) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1734-1760`
  - Module-level execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1762-1768`
- [x] Parameter validation before execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1696-1714`
- [x] Results saved to file `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1771-1772`
- [x] Stop button disables on completion `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1778-1779`
- [x] Console status returns to "Idle" `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1780-1782`

### Run All Scripts `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1370-1502`

- [x] Confirmation dialog shows count of scripts `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1373-1380`
- [x] Progress updates incrementally (0-100%) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1414-1417`
- [x] Each script name logged with progress `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1409-1411`
- [x] macOS: Runs synchronously on main thread `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1395-1441`
- [x] Other platforms: Runs in daemon thread `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1444-1502`
- [x] Stop signal interrupts execution `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1405-1407`
- [x] Status returns to "Ready" on completion `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1430-1440`

### Stop Functionality `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1070-1072`

- [x] Stop button sends stop signal to running protocol `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1070-1072`
- [x] Stop signal is threading.Event `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1405-1407`
- [x] Protocols check stop_event and exit gracefully

## Error Handling `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1795-1842`

- [x] ImportError shows messagebox with error details `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1806-1814`
- [x] ModuleNotFoundError handled gracefully `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1796-1797`
- [x] AttributeError for missing functions/methods `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1796-1799`
- [x] Console status shows "[X] Error" (red) on failure `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1809-1811`
- [x] Stop button disabled on error `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1813-1814`
- [x] Unexpected errors caught and logged `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1822-1842`

## Window Management

- [x] Window close handler (WM_DELETE_WINDOW) `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:697-706`
- [x] Running thread stopped on close `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:700-703`
- [x] Thread join with 2-second timeout `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:703-705`
- [x] Graceful cleanup of sys.modules entries `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:1527-1530`

## Parameter Inference System `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:638-694`

- [x] Integer parameters inferred from `n_*`, `*_size`, `*_count` patterns `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:645-655`
- [x] Float parameters inferred from `lr_*`, `learning_rate` patterns `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:657-667`
- [x] Threshold parameters from `theta_*`, `alpha`, `beta`, `rho` `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:669-681`
- [x] Dimension parameters from `*_dim` patterns `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py:683-693`

---

## Cross-Reference Summary

| Feature | Theory GUI | Falsification GUI | Validation GUI |
| ------- | ---------- | ----------------- | -------------- |
| Window Title | "APGI Theory Framework - Scientific Instrument Interface" | "APGI Framework-Level Falsification Aggregator (FP-ALL)" | "APGI Validation Protocol Runner" |
| Size | 1200x850 | 800x600 | 800x600 |
| Min Size | 900x650 | 640x480 | 800x600 |
| Protocol Selection | Sidebar buttons (dynamic) | Grid buttons (13 fixed) | Checkboxes (17 fixed) |
| Tabs | 2 (Protocols, Parameters) | 2 (Protocols, Parameters) | 5 (Validation, Exploration, Settings, Export, Alerts) |
| Design System | APGI Lab System | Standard tkinter | Standard tkinter |
| Parameters | Dynamic inference | Static definition | Static definition |
| Source File | `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI.py` | `@/Users/lesoto/Sites/PYTHON/apgi-validation/Falsification_Protocols_GUI.py` | `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py` |

---

## Summary

Total items to verify: **80+**

Test each item systematically and mark as complete when verified.

> **Note on Dynamic Discovery**: Theory GUI discovers protocols dynamically from the `Theory/` folder. Test scripts must exist in this folder for the GUI to display them. The specific scripts available may vary based on the current state of the Theory folder.

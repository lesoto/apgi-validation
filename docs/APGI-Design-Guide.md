# APGI Universal Application Design Guide

Standardized Interface Specifications for the APGI Software Ecosystem

---

## Table of Contents

1. [Universal Design Philosophy](#1-universal-design-philosophy)
2. [Technical Framework](#2-technical-framework)
3. [Visual Identity](#3-visual-identity)
4. [UI Components & Edge Cases](#4-ui-components--edge-cases)
5. [Universal Component Class](#5-universal-component-class)
6. [Layout Specifications](#6-layout-specifications)
7. [Accessibility Checklist](#7-accessibility-checklist)
8. [File References](#8-file-references)

---

## 1. Universal Design Philosophy

All APGI software, from the Experiments Suite to the Simulation Environment, must adhere to the **"Scientific Instrument"** aesthetic.

### Core Principles

| Principle                    | Description                                                                                                                                              |
|:-----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Instrument, Not App**      | The UI should feel like a medical dashboard or a laboratory workstation                                                                                  |
| **High Information Density** | Academic users prefer seeing more data at once over "simplified" consumer whitespace                                                                     |
| **Progressive Disclosure**   | Hide advanced mathematical guardrails (e.g., $B_t$ thresholds) behind collapsible frames or tabs                                                         |
| **The Intervention Rule**    | Every display of a neurological "deficit" or "state" (e.g., High Entropy) must be accompanied by a suggested intervention or action within the interface |

---

## 2. Technical Framework

### Standard Tkinter + ttk

To ensure **zero-dependency deployment** across research environments, all APGI Python applications use standard `tkinter` and `tkinter.ttk` (included in Python standard library).

### Theme Configuration

Since standard Tkinter can look dated, all apps must initialize with the `clam` or `alt` theme and manual color overrides to match the APGI palette.

```python
import tkinter as tk
from tkinter import ttk

def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use('clam')
    
    # Core Palette
    bg_color = "#f8f9fa"
    fg_color = "#212529"
    accent_blue = "#2874a6"
    border_color = "#dee2e6"
    
    # Configure Global Elements
    style.configure("TFrame", background=bg_color)
    style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Noto Sans", 10))
    style.configure("Header.TLabel", font=("Noto Sans", 12, "bold"))
    
    # Custom Card Style
    style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")
    
    # Button Styling
    style.configure("TButton", padding=6, background="#e9ecef")
    style.map("TButton",
              background=[('active', '#dee2e6'), ('disabled', '#f1f3f5')],
              foreground=[('disabled', '#adb5bd')])
              
    # Primary Button
    style.configure("Primary.TButton", background="#155724", foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map("Primary.TButton", background=[("active", "#0f3d1a")], foreground=[("active", "white")])
    
    # Secondary Button
    style.configure("Secondary.TButton", background="#2874a6", foreground="white", font=("Noto Sans", 10), padding=6)
    style.map("Secondary.TButton", background=[("active", "#1f5a82")], foreground=[("active", "white")])
    
    # Danger Button
    style.configure("Danger.TButton", background="#721c24", foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map("Danger.TButton", background=[("active", "#5a161d")], foreground=[("active", "white")])
    
    # Checkbutton
    style.configure("Card.TCheckbutton", background="#ffffff")
    
    # Configure root window
    root.configure(background=bg_color)
    
    return style
```

---

## 3. Visual Identity

### Color Palette (The "Lab" System)

| Category           | Hex Code  | Usage                                       |
|:-------------------|:----------|:--------------------------------------------|
| **Primary**        | `#2874a6` | Headers, primary buttons, active state      |
| **Success**        | `#155724` | Validated hypotheses, completed simulations |
| **Alert**          | `#721c24` | Prediction error, model divergence          |
| **Background**     | `#f8f9fa` | Main workspace background                   |
| **Surface**        | `#ffffff` | Individual cards, data entry fields         |
| **Border**         | `#dee2e6` | Dividers, card borders                      |
| **Text Primary**   | `#212529` | Headlines, important content                |
| **Text Secondary** | `#6c757d` | Labels, descriptions                        |

### Typography

| Type                         | Font           | Usage                                |
|:-----------------------------|:---------------|:-------------------------------------|
| **Primary (Digital)**        | Noto Sans      | Global standard for UI elements      |
| **Academic (Documentation)** | Noto Serif     | Research papers, citations           |
| **Monospace (Data/Code)**    | Noto Sans Mono | $B_t$ values, raw logs, code         |

---

## 4. UI Components & Edge Cases

### Status Indicators

| Status          | Visual Style | Logic                                             |
|:----------------|:-------------|:--------------------------------------------------|
| **Calibrating** | Blue Pulse   | Initial model alignment phase                     |
| **Inference**   | Marquee/Spin | Active data processing or API query               |
| **Gated**       | Static Gray  | Parameter $B_t$ threshold not yet reached         |
| **Success**     | Green + Icon | Process completed successfully (WCAG: must use ✔) |
| **Failed**      | Red + Icon   | Error encountered (WCAG: must use ✖)              |

### Empty States

When a data view is empty (e.g., no active experiments), do not show a blank screen. Display a centered placeholder:

- **Text:** "No active hypotheses found. Initialize a new simulation environment to begin."
- **Visual:** Use a `#e9ecef` (Soft Gray) outline or skeleton screen.

```python
# Empty state implementation
def create_empty_state(parent, message):
    frame = ttk.Frame(parent, padding=40)
    frame.pack(expand=True)
    
    # Soft gray outline
    canvas = tk.Canvas(frame, width=200, height=120, 
                       bg="#f8f9fa", highlightbackground="#e9ecef",
                       highlightthickness=2)
    canvas.pack()
    
    # Message
    label = ttk.Label(frame, text=message, wraplength=300,
                      font=("Noto Sans", 11), foreground="#6c757d")
    label.pack(pady=(20, 0))
    
    return frame
```

---

## 5. Universal Component Class

### APGICard — Standardized Information Card

This pattern should be used for all dashboard-style elements in the APGI lineup.

```python
import tkinter as tk
from tkinter import ttk

class APGICard(ttk.Frame):
    """Standardized information card for all APGI apps.
    
    Features:
    - Title with uppercase styling
    - Monospace value display
    - Optional intervention hint (The Intervention Rule)
    - Consistent padding and borders
    """
    
    def __init__(self, parent, title, value, intervention="", **kwargs):
        # Use custom style 'Card.TFrame' defined in apply_apgi_theme
        super().__init__(parent, style="Card.TFrame", **kwargs)
        
        # Internal padding via sub-frame
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)
        
        # Title (uppercase per lab convention)
        self.lbl_title = ttk.Label(
            container, 
            text=title.upper(), 
            style="Header.TLabel"
        )
        self.lbl_title.pack(anchor="w")
        
        # Value (monospace for scientific precision)
        self.lbl_value = ttk.Label(
            container, 
            text=value, 
            font=("Noto Sans Mono", 14)
        )
        self.lbl_value.pack(anchor="w", pady=(5, 10))
        
        # Intervention Rule (MANDATORY if displaying deficits)
        if intervention:
            separator = ttk.Separator(container, orient="horizontal")
            separator.pack(fill="x", pady=5)
            
            self.lbl_hint = ttk.Label(
                container, 
                text=f"Intervention: {intervention}",
                wraplength=250,
                foreground="#495057",
                font=("Noto Sans", 9, "italic")
            )
            self.lbl_hint.pack(anchor="w")

# Example Usage:
# card = APGICard(root, "Interoceptive Entropy", "0.84 H", "Increase grounding frequency")
# card.pack(padx=10, pady=10)
```

### Button Components

```python
class APGIButtons:
    """Standard button configurations for APGI applications."""
    
    @staticmethod
    def primary(parent, text, command):
        """Primary action button (green)."""
        return ttk.Button(parent, text=text, command=command, style="Primary.TButton", cursor="hand2")
    
    @staticmethod
    def danger(parent, text, command):
        """Danger/Stop button (red)."""
        return ttk.Button(parent, text=text, command=command, style="Danger.TButton", cursor="hand2")
    
    @staticmethod
    def secondary(parent, text, command):
        """Secondary action button (blue)."""
        return ttk.Button(parent, text=text, command=command, style="Secondary.TButton", cursor="hand2")
```

---

## 6. Layout Specifications

### Window Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  TOP METRIC BAR ($B_t$, epoch, connection status)          │
├────────────────┬──────────────────────────────────────────────┤
│                │                                              │
│   LEFT         │          WORKSPACE                         │
│   SIDEBAR      │          (graphs, tables, cards)           │
│   200px fixed  │                                              │
│                │                                              │
├────────────────┴──────────────────────────────────────────────┤
│  LOG CONSOLE (collapsible, real-time data stream)          │
└─────────────────────────────────────────────────────────────┘
```

### Grid Configuration

```python
def setup_layout(root):
    """Standard APGI application layout."""
    # Configure grid weights
    root.grid_columnconfigure(0, weight=0, minsize=200)  # Sidebar
    root.grid_columnconfigure(1, weight=1)                # Workspace
    root.grid_rowconfigure(0, weight=0)                   # Metric bar
    root.grid_rowconfigure(1, weight=1)                   # Main content
    root.grid_rowconfigure(2, weight=0, minsize=150)       # Console
```

### Spacing Scale

| Token | Pixels | Usage                 |
|:------|:-------|:----------------------|
| `xs`  | 4px    | Tight internal gaps   |
| `sm`  | 8px    | Component padding     |
| `md`  | 15px   | Card internal spacing |
| `lg`  | 20px   | Section separation    |
| `xl`  | 30px   | Window edge padding   |

---

## 7. Accessibility Checklist

### WCAG 2.1 AA Compliance

- [ ] **No Color-Only Meaning**: All error states have icons (✖/✔)
- [ ] **Contrast**: Text-to-background ratio is at least 4.5:1
- [ ] **Scale**: App layout maintains functionality at 125% OS-level scaling
- [ ] **Keyboard**: All navigation (Tabs, Buttons) accessible via Tab and Return

### Implementation Notes

```python
# Always include icon + color for status
def show_status(parent, status_type, message):
    icons = {
        "success": "✔",
        "error": "✖",
        "warning": "⚠",
        "info": "ℹ"
    }
    colors = {
        "success": "#155724",
        "error": "#721c24",
        "warning": "#856404",
        "info": "#2874a6"
    }
    
    label = ttk.Label(
        parent,
        text=f"{icons[status_type]} {message}",
        foreground=colors[status_type],
        font=("Noto Sans", 10, "bold")
    )
    return label
```

---

## 8. File References

| File                              | Purpose                                         |
|:----------------------------------|:------------------------------------------------|
| `GUI_auto_improve_experiments.py` | Reference implementation (uses CustomTkinter)   |
| `apgi-design-guide.md`            | This document — standard tkinter specifications |
| `design-guide.md`                 | Dark mode CustomTkinter variant                 |
| `app-design-guide.md`             | Light mode CustomTkinter variant                |

### Migration Path

To convert from CustomTkinter to standard tkinter:

1. Replace `ctk.CTk` with `tk.Tk`
2. Replace `ctk.CTkFrame` with `ttk.Frame` + `Card.TFrame` style
3. Replace `ctk.CTkButton` with `tk.Button` + color configuration
4. Replace `ctk.CTkLabel` with `ttk.Label`
5. Call `apply_apgi_theme(root)` at startup

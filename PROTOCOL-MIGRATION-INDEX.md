# Protocol Migration Documentation Index

**Master Index for Protocol Legacy-to-Canonical Migration**

---

## 📚 Documentation Files

### 1. **DISCOVERY-COMPLETE.md** (START HERE ⭐)
**Length**: ~600 lines | **Type**: Executive Summary | **Audience**: Everyone

Overview of the complete discovery and current status. Best starting point for understanding what was done, what works, and what's needed next.

**Contains**:
- What was discovered (47 scripts, 46 protocol files)
- Files created (4 documentation sets)
- Key findings (canonical files exist, legacy coexist, no breaking changes)
- What's working (protocol loader, Python scripts)
- What needs attention (Phase 2-4 over 6+ months)
- Immediate next steps

**Read this if**: You want a quick understanding of the full discovery in 10 minutes.

---

### 2. **PROTOCOL-QUICK-REFERENCE.md** (USE THIS DAILY 📋)
**Length**: ~900 lines | **Type**: Lookup Table & Reference | **Audience**: Developers

Comprehensive quick lookup for finding any protocol, its files, and how to use it.

**Contains**:
- All 47 scripts in quick lookup table
- Canonical APGI-P## protocols (8 files)
- Non-canonical VP/FP protocols (15+23 files)
- Master aggregators and runners
- Configuration files
- Protocol tiers and classifications
- 14 core named predictions
- Quick command reference

**Read this if**: You need to find a specific protocol or configuration quickly.

**Use cases**:
```
"What's the new file for VP-01?"
→ Go to "Canonical Protocols" table → APGI-P01 → protocol_1_cardiac_eeg.json

"Where's the Python script for APGI-P03?"
→ Go to "Canonical Protocols" table → APGI-P03 → VP_14_*, VP_15_*, FP_14_*

"How do I run all validation protocols?"
→ Go to "Quick Command Reference" → python -m Validation.Master_Validation
```

---

### 3. **docs/PROTOCOL-LEGACY-MAPPING.md** (DEFINITIVE REFERENCE 📖)
**Length**: ~3,000 lines | **Type**: Complete Technical Reference | **Audience**: Architects, Maintainers

Comprehensive mapping of every single protocol, file, and relationship. Definitive source of truth.

**Contains**:
- Executive summary with current state
- Complete protocol file mapping table (new → legacy → Python)
- All 22 VP scripts + metadata
- All 15 FP scripts + metadata
- Protocol linking (which protocols work together)
- Script usage patterns
- Loading patterns (4 different methods)
- Configuration files (all 10+)
- Named predictions registry
- Suggested deprecation timeline

**Read this if**: You need detailed information about specific protocols or the full reference.

**Use cases**:
```
"What's the dependency graph for APGI-P01?"
→ Find "APGI-P01: Cardiac EEG" section → see linked VP and FP protocols

"Which falsification criteria test APGI-P02?"
→ Find "APGI-P02: Somatic Agent Simulation" → see F1.x, F2.x criteria

"What are all the files for TMS protocol?"
→ Find "APGI-P05: Causal TMS" → lists all legacy files and Python implementations
```

---

### 4. **PROTOCOL-MIGRATION-GUIDE.md** (IMPLEMENTATION PLAN 🗺️)
**Length**: ~1,500 lines | **Type**: Implementation Roadmap | **Audience**: Project Leads, DevOps

Step-by-step migration plan with phases, timelines, and CLI commands.

**Contains**:
- What needs NO changes (Python scripts, loader logic, aggregators)
- What DOES need changes (after deprecation period)
- 4-phase migration plan:
  - Phase 1: Current (coexistence) - 0-1 month
  - Phase 2: Migration Notice - 1-3 months
  - Phase 3: Soft Deprecation - 3-6 months
  - Phase 4: Hard Deprecation - 6+ months
- Detailed updates needed for each component
- Configuration file review
- GUI reference file changes
- Test file updates
- Documentation updates
- CI/CD pipeline review
- Archive commands (for Phase 4)
- Verification commands
- Rollback plan (if needed)
- Timeline summary table
- Appendix with full file structure

**Read this if**: You're planning the migration timeline or implementation.

**Use cases**:
```
"When should we add deprecation warnings?"
→ Phase 2 (1-3 months) - see specific instructions

"How do we archive legacy files?"
→ Phase 4 - see "Migration Commands" section

"What if issues arise?"
→ See "Rollback Plan" section for step-by-step recovery
```

---

### 5. **PROTOCOL-DISCOVERY-RESULTS.md** (DETAILED FINDINGS 🔍)
**Length**: ~2,500 lines | **Type**: Technical Analysis | **Audience**: Developers, Architects

Comprehensive results from the discovery process with detailed statistics and execution flows.

**Contains**:
- Executive summary of discovery scope
- Complete protocol mapping (all 8 canonical)
- Validation protocols registry (23 + aggregator)
- Falsification protocols registry (15 + 2 aggregators)
- Scripts by category (47 total)
- Protocol loading architecture (4 methods)
- Configuration files (all 10+)
- Named predictions registry (14 core)
- Dependency graph (visual and text)
- Protocol execution flow (2 patterns)
- Security & integrity verification
- Statistics by category/tier/status
- Key files reference
- Action items timeline

**Read this if**: You want detailed technical findings from the discovery.

**Use cases**:
```
"What's the complete protocol execution flow?"
→ See "Protocol Execution Flow" section with step-by-step process

"How many protocols are in each tier?"
→ See "Statistics" section with breakdown by tier

"What are all the configuration files?"
→ See "Configuration Files" section with purpose of each

"How do protocols depend on each other?"
→ See "Dependency Graph" section with visual layout
```

---

## 🗂️ Which Document to Use?

```
Question                                Document              Section
──────────────────────────────────────────────────────────────────────
What was discovered?                   DISCOVERY-COMPLETE    "📋 What Was Done"
What's the quick status?                DISCOVERY-COMPLETE    "✅ What's Already Working"
How do I find a specific protocol?      QUICK-REFERENCE       "Protocol Lookup Table"
How do I load APGI-P01?                 QUICK-REFERENCE       "Protocol Load Methods"
What are all the 47 scripts?            QUICK-REFERENCE       "Master Aggregators & Runners"
Where's the Python file for VP-01?      LEGACY-MAPPING        "Validation Protocols"
What are all the falsification criteria? LEGACY-MAPPING        "Falsification Protocols"
How do protocols depend on each other?  DISCOVERY-RESULTS     "Dependency Graph"
When should we deprecate legacy files?  MIGRATION-GUIDE       "Deprecation Timeline"
How do I migrate the code?              MIGRATION-GUIDE       "Configuration File Updates"
What if something breaks?               MIGRATION-GUIDE       "Rollback Plan"
How many protocols are in each tier?    DISCOVERY-RESULTS     "Statistics"
What's the full execution flow?         DISCOVERY-RESULTS     "Protocol Execution Flow"
How do I verify the loader works?       MIGRATION-GUIDE       "Verify New Loader"
When do I archive legacy files?         MIGRATION-GUIDE       "Phase 4"
```

---

## 📊 Document Relationships

```
                    DISCOVERY-COMPLETE ⭐ (START HERE)
                           |
              ┌────────────┼────────────┬──────────────┐
              |            |            |              |
         Want detailed   Want quick   Want full       Want migration
         findings?       lookup?      reference?      plan?
              |            |            |              |
              ↓            ↓            ↓              ↓
      DISCOVERY-      QUICK-         LEGACY-        MIGRATION-
      RESULTS         REFERENCE      MAPPING        GUIDE
```

---

## 🎯 Reading Paths by Role

### For Project Manager
1. Read: **DISCOVERY-COMPLETE.md** (5 minutes)
2. Reference: **MIGRATION-GUIDE.md** timeline (5 minutes)
3. Share: **DISCOVERY-COMPLETE.md** with team

**Time**: 10 minutes

### For Developer
1. Read: **DISCOVERY-COMPLETE.md** (10 minutes)
2. Bookmark: **QUICK-REFERENCE.md** (daily use)
3. Reference: **LEGACY-MAPPING.md** when needed (questions)
4. Study: **DISCOVERY-RESULTS.md** for deep understanding (optional)

**Time**: 20 minutes + ongoing reference

### For DevOps/Infrastructure
1. Read: **DISCOVERY-COMPLETE.md** (5 minutes)
2. Study: **MIGRATION-GUIDE.md** in detail (30 minutes)
3. Reference: **QUICK-REFERENCE.md** CLI commands (ongoing)
4. Archive: Keep Phase 4 commands for future use

**Time**: 40 minutes + 6 months execution

### For Architect/Tech Lead
1. Read: **DISCOVERY-COMPLETE.md** (10 minutes)
2. Study: **LEGACY-MAPPING.md** (30 minutes)
3. Study: **DISCOVERY-RESULTS.md** (30 minutes)
4. Review: **MIGRATION-GUIDE.md** (20 minutes)

**Time**: 90 minutes for full understanding

---

## ✅ Checklist: Before You Start

- [ ] Read DISCOVERY-COMPLETE.md
- [ ] Skim QUICK-REFERENCE.md to see what's available
- [ ] Understand there are NO breaking changes required
- [ ] Know that both old and new naming schemes work together
- [ ] Recognize the 6-month migration timeline is gradual
- [ ] Have questions? Refer to appropriate document above

---

## 🚀 Next Steps

### Week 1
1. ✅ Read DISCOVERY-COMPLETE.md (everyone on team)
2. ✅ Run: `pytest tests/test_protocol_*.py -v` (verify nothing broke)
3. ✅ Test: `python -c "from utils.protocol_loader import load_protocol; print(load_protocol('APGI-P01'))"`

### Weeks 2-4
1. Review LEGACY-MAPPING.md for your specific protocols
2. Update inline code documentation with new naming
3. Brief team on migration timeline

### Month 2-3
1. Implement Phase 2 (migration notices)
2. Update user-facing documentation
3. Include in release notes

### Months 4-6
1. Implement Phase 3 (soft deprecation)
2. Monitor user feedback

### Month 6+
1. Implement Phase 4 (hard deprecation)
2. Archive legacy files

---

## 📞 FAQ

**Q: Do I need to change anything right now?**  
A: No. Both naming schemes work. See DISCOVERY-COMPLETE.md "What's Already Working"

**Q: Will this break my code?**  
A: No. This is a coexistence migration with no breaking changes.

**Q: When do legacy files get removed?**  
A: Phase 4 (6+ months). See MIGRATION-GUIDE.md "Timeline Summary"

**Q: How do I use the new protocol naming?**  
A: See QUICK-REFERENCE.md "Protocol Load Methods" section

**Q: Which protocol corresponds to what?**  
A: See QUICK-REFERENCE.md "Canonical Protocols" or LEGACY-MAPPING.md for full mapping

**Q: Is the loader already updated?**  
A: Yes. See DISCOVERY-COMPLETE.md "What's Already Working"

---

## 📄 File Locations

All documentation files are in the project root:

```
/Users/lesoto/Sites/PYTHON/apgi-validation/
├── DISCOVERY-COMPLETE.md                    ← Start here ⭐
├── PROTOCOL-MIGRATION-INDEX.md             ← This file
├── PROTOCOL-QUICK-REFERENCE.md             ← Quick lookup 📋
├── PROTOCOL-MIGRATION-GUIDE.md             ← Implementation 🗺️
├── PROTOCOL-DISCOVERY-RESULTS.md           ← Detailed findings 🔍
├── docs/
│   └── PROTOCOL-LEGACY-MAPPING.md          ← Full reference ��
└── (all other project files)
```

---

## 🎓 TL;DR

- **Discovery**: ✅ Complete (all 47 scripts, 46 protocol files mapped)
- **Status**: ✅ Safe (no breaking changes, both schemes work)
- **Timeline**: ✅ Gradual (6+ months, 4 phases)
- **Action**: Read DISCOVERY-COMPLETE.md, then QUICK-REFERENCE.md
- **Next Step**: Run tests, verify loader works
- **Timeline Start**: 6 months to Phase 4 completion

---

**Happy coding! And remember: both `protocol_vp_01.json` and `protocol_1_cardiac_eeg.json` work right now.** ✨

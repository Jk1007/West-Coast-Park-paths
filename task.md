# Tasks: Public Mobile View

- [x] Create Proposal (`implementation_plan.md`) <!-- id: 0 -->
    - [x] Define `/public` page layout <!-- id: 1 -->
    - [x] Define "Simulated User" behavior for demo <!-- id: 2 -->
    - [x] Define Admin -> Public navigation <!-- id: 3 -->
- [x] Refactor `wcp_nicegui.py` for multi-page support <!-- id: 4 -->
    - [x] specific `admin_page` function for existing UI <!-- id: 5 -->
    - [x] `public_page` function for new UI <!-- id: 6 -->
- [x] Implement Public View Logic <!-- id: 7 -->
    - [x] Show map (read-only) <!-- id: 8 -->
    - [x] Display "You are Here" marker <!-- id: 9 -->
    - [x] Calculate and draw orange evacuation path <!-- id: 10 -->
- [x] Add "Open Public View" button to Admin panel <!-- id: 11 -->
- [x] Verify functionality <!-- id: 12 -->

# Tasks: OWASP Security Hardening

- [x] Audit Codebase for Vulnerabilities <!-- id: 13 -->
    - [x] Check for hardcoded secrets (API keys) in `wcp_weather.py` <!-- id: 14 -->
    - [x] Check for XSS in UI inputs (`location_input`) <!-- id: 15 -->
    - [x] Check for Insecure deserialization or unsafe imports <!-- id: 16 -->
- [x] Implement Vulnerability Fixes <!-- id: 17 -->
    - [x] Sanitize user inputs in `wcp_nicegui.py` and `wcp_core.py` <!-- id: 18 -->
    - [x] Externalize secrets (if any) using env vars or placeholder <!-- id: 19 -->
    - [x] Add basic rate limiting or input length constraints <!-- id: 20 -->
    - [x] Ensure `ui.run` has secure defaults (no `reload=True` in prod, localhost binding) <!-- id: 21 -->
    - [x] Ensure `.env` is ignored by git (`.gitignore`) <!-- id: 23 -->
- [ ] Verify functionality after hardening <!-- id: 22 -->

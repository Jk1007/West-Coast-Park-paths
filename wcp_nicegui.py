# wcp_nicegui.py
# NiceGUI frontend that wraps the West Coast Park core sim

# NiceGUI frontend that wraps the West Coast Park core sim

from nicegui import ui, app  # Added 'app' for storage
import os
import html
import secrets  # <--- For dynamic session keys

# --- Manual .env loader (No hardcoded secrets) ---
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k] = v
except FileNotFoundError:
    print("Warning: .env file not found. Admin login may fail.")

from wcp_core import (  # imports simulation + chart
    tick, running, PEOPLE, HAZARDS,
    hazard_radius,
    reset_model, _recompute_safe_nodes, _recompute_featured_safe,
    _choose_targets_and_paths, _force_evacuation_mode, _optimize_safe_zones_by_eta,
    _point_from_phrase, kpi_eta_summary,
    _totals_now, park_chart, EVAC_NOTIFICATION_TICK,
    add_hazard_at, remove_hazard_near)
import wcp_responders as responders

from datetime import datetime
from wcp_weather import weather_now_str, forecast_2h_str, forecast_24h_str
import wcp_weather as weather
import numpy as np
import plotly.graph_objects as go
import asyncio
import wcp_core as core
from wcp_core import time_str, date_str
import wcp_nicegui_mobile as mobile_ui  # <--- NEW MODULE

# Prevent timer_tick from stepping/redrawing while we mutate hazards/safe zones
ui_busy = False
_started_weather = False


# ---------- helpers to keep labels in sync ----------

def update_labels(status_label, eta_label):
    t = _totals_now()
    status_label.text = (
        f"Tick: {int(tick.value)} | "
        f"Total: {t['total']} | In-envelope: {t['evacuees']} | "
        f"Exposed to either C,B,R or E: {t['exposed']} | Aware: {t['aware']} | Reached: {t['reached']}"
    )
    eta_label.text = kpi_eta_summary()


def redraw(plot):
    """Re-render Plotly figure from the core."""
    plot.figure = park_chart()
    plot.update()


# ---------- NiceGUI layout (Admin Page) ----------

@ui.page('/')
def admin_page():
    # --- OWASP A01: Broken Access Control (Admin Login) ---
    def try_login():
        if pwd_input.value == os.environ.get("ADMIN_PASSWORD"):
            app.storage.user['authenticated'] = True
            ui.navigate.to('/')
        else:
            ui.notify("Invalid Password", type='negative')

    if not app.storage.user.get('authenticated', False):
        with ui.column().classes('w-full h-screen items-center justify-center bg-gray-900 text-white'):
            ui.label('Restricted Access').classes('text-2xl font-bold mb-4')
            pwd_input = ui.input('Admin Password', password=True).on('keydown.enter', try_login).classes('w-64 bg-gray-800')
            ui.button('Login', on_click=try_login).classes('w-64 bg-blue-600 mt-2')
        return

    global ui_busy
    with ui.row().classes('w-full h-screen no-wrap gap-0'):

        # ===== LEFT PANEL =====
        with ui.column().classes('w-[30%] p-2 gap-1'):

            ui.markdown('## West Coast Park (Evac)')

            # --- Card 1: Environment ---
            with ui.card().classes('w-full p-2 gap-1 bg-gray-100'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Environment').classes('font-bold text-xs')
                    with ui.row().classes('gap-2 text-xs'):
                        ui.label().bind_text_from(time_str, 'value').classes('font-mono')
                        ui.label().bind_text_from(date_str, 'value').classes('font-mono')
                
                # Weather Column (Vertical Stack)
                with ui.column().classes('w-full gap-0 text-xs'):
                    ui.label('Current Weather:').classes('font-bold text-gray-800 mt-1')
                    ui.label().bind_text_from(weather_now_str, 'value').classes('ml-2')

                    ui.label('2-Hour Forecast:').classes('font-bold text-gray-800 mt-1')
                    ui.label().bind_text_from(forecast_2h_str, 'value').classes('ml-2 text-gray-600')

                    ui.label('24-Hour Forecast:').classes('font-bold text-gray-800 mt-1')
                    ui.label().bind_text_from(forecast_24h_str, 'value').classes('ml-2 text-gray-500 italic')

                    ui.label().bind_text_from(weather.last_error, 'value').classes('text-red font-bold')

            # --- Card 2: Simulation Status ---
            with ui.card().classes('w-full p-2 gap-1 bg-blue-50'):
                ui.label('Status').classes('font-bold text-xs mb-1')
                status_label = ui.label().classes('text-xs font-mono whitespace-pre-wrap')
                eta_label = ui.label().classes('text-xs font-mono')

            # --- STEP / RUN / RESET ---

            def do_step():
                from wcp_core import _step_once   # avoid circular import at top
                _step_once()
                update_labels(status_label, eta_label)
                redraw(plot)

            def toggle_run():
                running.value = not running.value
                btn_run.text = 'Pause' if running.value else 'Run'

            def reset_all():
                running.value = False
                reset_model()
                _recompute_safe_nodes()
                _recompute_featured_safe()
                _choose_targets_and_paths()
                update_labels(status_label, eta_label)
                redraw(plot)
                btn_run.text = 'Run'

            with ui.row().classes('w-full gap-2'):
                ui.button('STEP', on_click=do_step).classes('grow')
                btn_run = ui.button('Run', on_click=toggle_run).classes('grow')
                ui.button('RESET', on_click=reset_all).classes('grow')

            # --- Hazards ---

            ui.markdown('### Hazards')
            location_input = ui.input(
                'Where is the incident?',
                placeholder='e.g. North, NE, 160°, 225°, 300m from center',
            ).classes('w-full')
            location_input.on('keydown.enter', lambda e: add_hazard())


                    # list all hazards by ID and description (where)
            hazard_select = ui.select({},
                label='Existing hazards (ID — location)',
            ).classes('w-full')

            def refresh_hazard_list():
                """Update the hazard dropdown with all hazards."""
                options = {}

                for h in HAZARDS:
                    hid = h['id']
                    label = h.get('label')
                    if not label:
                        pos = h['pos']
                        label = f"({pos[0]:.0f}, {pos[1]:.0f})"
                    options[str(hid)] = f"#{hid} — {label}"
                
                # Force NiceGUI to rebuild dropdown
                hazard_select.set_options(options)

                # Auto-select first hazard if list not empty
                if options:
                    if hazard_select.value not in options:
                        hazard_select.value = next(iter(options.keys()))
                else:
                    hazard_select.value = None

            # Timer: continuous sim while running.value is True
            def add_hazard():
                import wcp_core as core
                import traceback
                global ui_busy

                if ui_busy:
                    ui.notify('Busy updating map, try again in 1s.', type='warning')
                    return

                ui_busy = True
                was_running = bool(core.running.value)
                core.running.value = False

                try:
                    # OWASP A03: Injection (Input Limit)
                    phrase = (location_input.value or '').strip()
                    if len(phrase) > 100:
                        phrase = phrase[:100]

                    pt = _point_from_phrase(phrase)

                    if pt is None:
                        ui.notify(
                            'Could not parse location. Try: North, South, East, West, NE/NW/SE/SW, or "160°, 300m".',
                            type='warning',
                        )
                        return

                    hx, hy = float(pt[0]), float(pt[1])
                    # OWASP A03: Injection (XSS Prevention)
                    safe_phrase = html.escape(phrase)
                    label = safe_phrase if safe_phrase else '(' + str(round(hx, 0)) + ', ' + str(round(hy, 0)) + ')'

                    core.HAZARDS.append({
                        'id': core.get_next_hazard_id_str(),
                        'pos': np.array([hx, hy], dtype=float),
                        'origin_pos': np.array([hx, hy], dtype=float),  # keep arrows stable
                        'r_m': float(max(5.0, hazard_radius.value)),
                        'label': label,
                        'created_at': datetime.now().strftime("%d %b %Y, %H:%M:%S"),
                        'type': "N/A"
                    })

                    _recompute_safe_nodes()
                    _recompute_featured_safe()
                    _choose_targets_and_paths()

                    EVAC_NOTIFICATION_TICK.value = int(tick.value)
                    for p in PEOPLE:
                        p['evac_start_tick'] = int(tick.value)
                        p['evac_end_tick'] = None
                        p['evac_time_s'] = None

                    _force_evacuation_mode()

                    update_labels(status_label, eta_label)
                    redraw(plot)
                    refresh_hazard_list()

                    ui.notify('Hazard added (ID ' + str(core.HAZARD_ID - 1) + '), agents evacuating.', type='positive')

                except Exception as e:
                    print('Error in add_hazard:', e)
                    traceback.print_exc()
                    # OWASP A05: Security Misconfiguration (Mask Errors)
                    ui.notify('Add hazard failed (Invalid Input)', type='negative')

                finally:
                    if was_running:
                        core.running.value = True
                    ui_busy = False

            def clear_hazards():
                global ui_busy
                ui_busy = True
                HAZARDS.clear()
                _recompute_safe_nodes()
                _recompute_featured_safe()
                _choose_targets_and_paths()
                update_labels(status_label, eta_label)
                redraw(plot)
                refresh_hazard_list()
                ui_busy = False
                ui.notify('All hazards cleared.', type='positive')

            def remove_last_hazard():
                global ui_busy
                if ui_busy:
                    ui.notify('Busy updating map, try again in 1s.', type='warning')
                    return

                ui_busy = True
                try:
                    if not HAZARDS:
                        ui.notify('No hazards to remove.', type='warning')
                        return

                    hid = HAZARDS[-1]['id']
                    HAZARDS.pop()

                    _recompute_safe_nodes()
                    _recompute_featured_safe()
                    _choose_targets_and_paths()

                    update_labels(status_label, eta_label)
                    redraw(plot)
                    refresh_hazard_list()

                    ui.notify('Removed last hazard (ID ' + str(hid) + ').', type='positive')
                finally:
                    ui_busy = False


            
            def remove_selected_hazard():
                global ui_busy
                if ui_busy:
                    ui.notify('Busy updating map, try again in 1s.', type='warning')
                    return

                ui_busy = True
                try:
                    if not hazard_select.value:
                        ui.notify('No hazard selected.', type='warning')
                        return

                    target_id = hazard_select.value
                    found = False
                    for i, h in enumerate(HAZARDS):
                        if str(h.get('id')) == str(target_id):
                            HAZARDS.pop(i)
                            found = True
                            break

                    if not found:
                        ui.notify('Hazard ID ' + str(target_id) + ' not found.', type='warning')
                        return

                    _recompute_safe_nodes()
                    _recompute_featured_safe()
                    _choose_targets_and_paths()

                    update_labels(status_label, eta_label)
                    redraw(plot)
                    refresh_hazard_list()

                    ui.notify('Removed hazard ID ' + str(target_id) + '.', type='positive')
                finally:
                    ui_busy = False


            with ui.row().classes('w-full gap-2'):
                ui.button('Add Hazard', on_click=add_hazard).classes('grow')
                ui.button('Clear Hazards', on_click=clear_hazards).classes('grow')

            with ui.row().classes('w-full gap-2'):
                ui.button('Remove Last Hazard', on_click=remove_last_hazard).classes('grow')
                ui.button('Remove Selected', on_click=remove_selected_hazard).classes('grow')

            # in case there are pre-existing hazards from the core:
            refresh_hazard_list()

            # --- Safe zones & SCDF responders ---

            ui.markdown('### Safe Zones & SCDF') 
            

            def optimise_safe_eta():
                try:
                    # use core defaults: k from N_SAFE, speed_mps=1.4
                    ok = _optimize_safe_zones_by_eta(
                        attempts=10,        # optional tuning
                        sample_size=500,    # optional tuning
                    )

                    _recompute_featured_safe()
                    _choose_targets_and_paths()
                    update_labels(status_label, eta_label)
                    redraw(plot)

                    ui.notify(
                        'Optimised safe zones for min ETA.' if ok else 'Safe-zone optimisation failed.',
                        type='positive' if ok else 'warning',
                    )

                except Exception as e:
                    import traceback
                    print('Error in optimise_safe_eta:', e)
                    traceback.print_exc()
                    ui.notify(f'Error optimising safe zones: {e}', type='negative')



            def suggest_responders():
                try:
                    # let core decide how many responders to use
                    # Pass all dependencies explicitly to the new module
                    ok = responders.suggest_responders(
                        hazards=core.HAZARDS,
                        people=core.PEOPLE,
                        kd_tree=core.KD,
                        node_ids=core.NODE_IDS,
                        pos_dict=core.POS_DICT,
                        map_dist_func=core.map_distance_m
                    )

                    update_labels(status_label, eta_label)
                    redraw(plot)

                    ui.notify(
                        'Suggested SCDF responder staging points.' if ok else 'Responder suggestion failed.',
                        type='positive' if ok else 'warning',
                    )

                except Exception as e:
                    import traceback
                    print('Error in suggest_responders:', e)
                    traceback.print_exc()
                    ui.notify(f'Error suggesting SCDF responders: {e}', type='negative')



            with ui.row().classes('w-full gap-2'):
                ui.button('Optimise Safe Zones (min ETA)', on_click=optimise_safe_eta).classes('w-full')

            with ui.row().classes('w-full gap-2'):
                ui.button('Suggest SCDF Responders', on_click=suggest_responders).classes('grow')
                ui.button(
                    'Clear Responders',
                    on_click=lambda: (responders.RESPONDERS.clear(), redraw(plot), ui.notify('Cleared responders.', type='positive')),
                ).classes('grow')
            
            ui.markdown('### Public View')
            ui.button('Open Public View (Mobile)', on_click=lambda: ui.navigate.to('/public', new_tab=True)).classes('w-full bg-purple-600 text-white')

            # Initialise labels once
            update_labels(status_label, eta_label)


            def start_weather_once():
                global _started_weather
                if _started_weather:
                    return
                _started_weather = True

                asyncio.create_task(weather.weather_loop())
                asyncio.create_task(core.time_date_loop())

            ui.timer(1.0, start_weather_once, once=True)
            def timer_tick():
                global ui_busy
                if ui_busy:
                    return
                if not running.value:
                    return

                from wcp_core import _step_once  # import here to avoid circular import issues
                _step_once()
                update_labels(status_label, eta_label)
                redraw(plot)

            ui.timer(0.12, timer_tick)




        # ===== RIGHT PANEL – plot =====
        with ui.column().classes('w-[70%] p-0 gap-0'):
            plot = ui.plotly(park_chart()).classes('w-full h-full')
        
        # --- Map Click Interaction ---
        click_coords = {"x": 0, "y": 0}
        with ui.dialog() as hazard_config_dialog, ui.card():
            ui.label('Map Options').classes('text-lg font-bold')
            
            def on_create_click():
                add_hazard_at(click_coords["x"], click_coords["y"])
                hazard_config_dialog.close()
                refresh_hazard_list()
                redraw(plot)
                
            def on_remove_click():
                remove_hazard_near(click_coords["x"], click_coords["y"])
                hazard_config_dialog.close()
                refresh_hazard_list()
                redraw(plot)

            ui.button('Create Hazard Here', on_click=on_create_click).classes('w-full color-red')
            ui.button('Remove Nearest Hazard', on_click=on_remove_click).classes('w-full')
            ui.button('Cancel', on_click=hazard_config_dialog.close).classes('w-full flat')

        def handle_map_click(e):
            points = e.args.get('points')
            if points:
                click_coords["x"] = points[0]['x']
                click_coords["y"] = points[0]['y']
                ui.notify(f"Selected: {click_coords['x']:.0f}, {click_coords['y']:.0f}")
                hazard_config_dialog.open()

        plot.on('plotly_click', handle_map_click, ['points'])
        redraw(plot)


# ---------- Register Mobile Page ----------
mobile_ui.init_mobile_page()

# ---------- run NiceGUI ----------

if __name__ == '__main__':
    ui.run(
        title='West Coast Park Evac (NiceGUI)',
        host='127.0.0.1',
        port=8080,
        reload=False,
        storage_secret=secrets.token_hex(32) # OWASP A07: Dynamic secret forces new login on restart
    )

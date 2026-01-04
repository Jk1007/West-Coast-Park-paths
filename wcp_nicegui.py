# wcp_nicegui.py
# NiceGUI frontend that wraps the West Coast Park core sim

from nicegui import ui
from wcp_core import (  # imports simulation + chart
    tick, running, PEOPLE, HAZARDS, RESPONDERS,
    hazard_radius,
    reset_model, _recompute_safe_nodes, _recompute_featured_safe,
    _choose_targets_and_paths, _force_evacuation_mode, _optimize_safe_zones_by_eta,
    _suggest_responders, _point_from_phrase, kpi_eta_summary,
    _totals_now, park_chart, EVAC_NOTIFICATION_TICK,
    add_hazard_at, remove_hazard_near)

from datetime import datetime
from wcp_weather import weather_now_str, forecast_2h_str, forecast_24h_str
import wcp_weather as weather
import numpy as np
import asyncio
import wcp_core as core
from wcp_core import time_str, date_str
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


# ---------- NiceGUI layout ----------

with ui.row().classes('w-full h-screen'):

    # ===== LEFT PANEL =====
    with ui.column().classes('w-1/3 p-4 gap-2'):

        ui.markdown('## West Coast Park – Evacuation & SCDF (NiceGUI)')

        status_label = ui.label()
        eta_label = ui.label()

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
            ui.button('STEP', on_click=do_step).classes('w-1/3')
            btn_run = ui.button('Run', on_click=toggle_run).classes('w-1/3')
            ui.button('RESET', on_click=reset_all).classes('w-1/3')

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
                phrase = (location_input.value or '').strip()
                pt = _point_from_phrase(phrase)

                if pt is None:
                    ui.notify(
                        'Could not parse location. Try: North, South, East, West, NE/NW/SE/SW, or "160°, 300m".',
                        type='warning',
                    )
                    return

                hx, hy = float(pt[0]), float(pt[1])
                label = phrase if phrase else '(' + str(round(hx, 0)) + ', ' + str(round(hy, 0)) + ')'

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
                ui.notify('Add hazard failed: ' + str(e), type='negative')

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
            ui.button('Add Hazard', on_click=add_hazard).classes('w-1/2')
            ui.button('Clear Hazards', on_click=clear_hazards).classes('w-1/2')
        ui.button('Remove Last Hazard', on_click=remove_last_hazard).classes('w-full mt-1')
        ui.button('Remove Selected Hazard', on_click=remove_selected_hazard).classes('w-full mt-1')

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
                ok = _suggest_responders()

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
            ui.button('Suggest SCDF Responders', on_click=suggest_responders).classes('w-1/2')
            ui.button(
                'Clear Responders',
                on_click=lambda: (RESPONDERS.clear(), redraw(plot), ui.notify('Cleared responders.', type='positive')),
            ).classes('w-1/2')

        ui.separator()
        ui.markdown('### Status')
        ui.markdown('**Weather (NEA)**')

        with ui.grid(columns=2).classes('w-full gap-2'):
            ui.label('Now')
            ui.label().bind_text_from(weather_now_str, 'value')

            ui.label('Next 2 hours')
            ui.label().bind_text_from(forecast_2h_str, 'value')

            ui.label('Next 24 hours')
            ui.label().bind_text_from(forecast_24h_str, 'value')
        #checking for error
        ui.label().bind_text_from(weather.last_error, 'value')

        ui.label().bind_text_from(time_str, 'value')
        ui.label().bind_text_from(date_str, 'value')

        ui.markdown('**Evacuation ETA summary:**')
        ui.label().bind_text_from(eta_label, 'text')
        ui.markdown('**Counts:**')
        ui.label().bind_text_from(status_label, 'text')

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
    with ui.column().classes('w-2/3 p-4'):
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


# ---------- run NiceGUI ----------

if __name__ == '__main__':
    ui.run(
        title='West Coast Park Evac (NiceGUI)',
        host='127.0.0.1',
        port=8080,
        reload=False,
    )

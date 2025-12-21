# wcp_nicegui.py
# NiceGUI frontend that wraps the West Coast Park core sim

from nicegui import ui
from wcp_core import (  # imports simulation + chart
    tick, running, PEOPLE, HAZARDS, RESPONDERS,
    hazard_radius,
    reset_model, _recompute_safe_nodes, _recompute_featured_safe,
    _choose_targets_and_paths, _force_evacuation_mode, _optimize_safe_zones_by_eta,
    _suggest_responders, _point_from_phrase, kpi_eta_summary,
    _totals_now, park_chart, EVAC_NOTIFICATION_TICK)

from wcp_weather import weather_now_str, forecast_2h_str, forecast_24h_str
import wcp_weather as weather
import numpy as np
import asyncio
import wcp_core as core
from wcp_core import time_str, date_str





# ---------- helpers to keep labels in sync ----------

def update_labels(status_label, eta_label):
    t = _totals_now()
    status_label.text = (
        f"Tick: {int(tick.value)} | "
        f"Total: {t['total']} | In-envelope: {t['evacuees']} | "
        f"Aware: {t['aware']} | Reached: {t['reached']}"
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

        def add_hazard():
            phrase = (location_input.value or '').strip()
            pt = _point_from_phrase(phrase)
            if pt is None:
                ui.notify(
                    'Could not parse location. Try: North, South, East, West, NE, NW, SE, SW, or "160°, 300m".',
                    type='warning',
                )
                return

            
            # NOTE: because we imported HAZARD_ID by value, we use the module directly
            import wcp_core as core

              # build a user-friendly label: use phrase, fallback to coords
            hx, hy = float(pt[0]), float(pt[1])
            label = phrase if phrase else f"({hx:.0f}, {hy:.0f})"

            core.HAZARDS.append({
                'id': core.HAZARD_ID,
                'pos': np.array([hx, hy], dtype=float),
                'r_m': float(max(5.0, hazard_radius.value)),
                'label': label,  # <-- this stores the "where"
            })
            core.HAZARD_ID += 1

            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()

            # everyone starts evacuating once hazard declared
            EVAC_NOTIFICATION_TICK.value = int(tick.value)
            for p in PEOPLE:
                p['evac_start_tick'] = int(tick.value)
                p['evac_end_tick'] = None
                p['evac_time_s'] = None
            _force_evacuation_mode()

            update_labels(status_label, eta_label)
            redraw(plot)
            refresh_hazard_list()
            ui.notify('Hazard added, agents evacuating.', type='positive')

        def clear_hazards():
            HAZARDS.clear()
            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            update_labels(status_label, eta_label)
            redraw(plot)
            refresh_hazard_list()
            ui.notify('All hazards cleared.', type='positive')

        def remove_last_hazard():
            if not HAZARDS:
                ui.notify('No hazards to remove.', type='warning')
                return
            hid = int(HAZARDS[-1]['id'])
            HAZARDS.pop()
            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            update_labels(status_label, eta_label)
            redraw(plot)
            refresh_hazard_list()
            ui.notify(f'Removed last hazard (ID {hid}).', type='positive')

        
        def remove_selected_hazard():
            if not hazard_select.value:
                ui.notify('No hazard selected.', type='warning')
                return
            target_id = int(hazard_select.value)

            # delete matching hazard from HAZARDS list
            for i, h in enumerate(HAZARDS):
                if int(h.get('id')) == target_id:
                    HAZARDS.pop(i)
                    break
            else:
                ui.notify(f'Hazard ID {target_id} not found.', type='warning')
                return

            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            update_labels(status_label, eta_label)
            redraw(plot)
            refresh_hazard_list()
            ui.notify(f'Removed hazard ID {target_id}.', type='positive')

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
            ui.label('Now').classes('font-bold')
            ui.label().bind_text_from(weather_now_str, 'value')

            ui.label('Next 2 hours').classes('font-bold')
            ui.label().bind_text_from(forecast_2h_str, 'value')

            ui.label('Next 24 hours').classes('font-bold')
            ui.label().bind_text_from(forecast_24h_str, 'value')
        ui.label().bind_text_from(time_str, 'value')
        ui.label().bind_text_from(date_str, 'value')
        ui.markdown('**Evacuation ETA summary:**')
        ui.label().bind_text_from(eta_label, 'text')
        ui.markdown('**Counts:**')
        ui.label().bind_text_from(status_label, 'text')

        # Initialise labels once
        update_labels(status_label, eta_label)

        # Timer: continuous sim while running.value is True
        def timer_tick():
            try:
                if running.value:
                    from wcp_core import _step_once
                    _step_once()
                    update_labels(status_label, eta_label)
                    redraw(plot)
            except Exception as e:
                import traceback
                print('Timer error:', e)
                traceback.print_exc()
                running.value = False
                ui.notify(f'Simulation error, paused: {e}', type='negative')

        ui.timer(0.2, timer_tick)  # 5 FPS
        # Start NEA background updater once (script-mode safe)
        _started_weather = False

        def start_weather_once():
            global _started_weather
            if _started_weather:
                return
            _started_weather = True
            asyncio.create_task(weather.weather_loop())

        ui.timer(1.0, start_weather_once, once=True)


    # ===== RIGHT PANEL – plot =====
    with ui.column().classes('w-2/3 p-4'):
        plot = ui.plotly(park_chart()).classes('w-full h-full')
        # first draw
        redraw(plot)


# ---------- run NiceGUI ----------

if __name__ == '__main__':
    ui.run(
        title='West Coast Park Evac (NiceGUI)',
        host='127.0.0.1',
        port=8080,
        reload=False,
    )
asyncio.create_task(core.time_date_loop())
from nicegui import ui, app
import numpy as np
import plotly.graph_objects as go
import wcp_core as core

def init_mobile_page():
    @ui.page('/public')
    def public_page():
        # OWASP A01: Strict Logout (Force re-login if user goes back to Admin)
        app.storage.user['authenticated'] = False
        # Simulated User Location (fixed for demo)
        user_pos = np.array([120, 300], dtype=float)

        # Outer container (background)
        with ui.column().classes('w-full h-screen items-center justify-center bg-gray-900'):
            
            # Phone Frame (Bezel, Shadow, Rounded)
            with ui.column().classes('w-[375px] h-[800px] bg-white border-[16px] border-black rounded-[3rem] overflow-hidden shadow-2xl relative p-0 gap-0'):
                
                # Header
                ui.label('West Coast Park (Public Evac)').classes('w-full text-center font-bold text-lg bg-gray-200 p-2')
                
                # Status Banner
                status_banner = ui.label('Loading...').classes('w-full text-center p-2 font-bold text-white')

                # Map Container
                # Config: scrollZoom=True allows pinching/scrolling. displayModeBar=False hides top toolbar.
                # Fixed 500 Error: ui.plotly does not support .config() chaining.
                plot = ui.plotly(core.park_chart()).classes('w-full grow')

            def update_public_view():
                # 1. Update Status Banner
                if core.HAZARDS:
                    status_banner.text = '⚠️ HAZARD DETECTED - PROCEED TO SAFE ZONE'
                    status_banner.classes(replace='bg-red-500 animate-pulse')
                else:
                    status_banner.text = 'Condition Normal - Enjoy the Park'
                    status_banner.classes(replace='bg-green-500')

                # 2. Calculate Path for Simulated User
                # Snap to graph
                _, start_node = core._nearest_node_idx(user_pos[0], user_pos[1])
                # Find target
                if core.HAZARDS:
                     target_node = core._nearest_safe_node_from(start_node)
                else:
                     target_node = start_node # No evac needed

                # Get path
                path_nodes = core._nx_path(start_node, target_node)
                path_x = [core.POS_DICT[n][0] for n in path_nodes]
                path_y = [core.POS_DICT[n][1] for n in path_nodes]

                # 3. Get fresh map & Add overlays
                fig = core.park_chart()
                
                # Fix Usability: Enable Pan, persistence (uirevision), and no margins
                fig.update_layout(
                    dragmode='pan',
                    hovermode='closest',
                    uirevision='constant',  # <--- CRITICAL: Preserves zoom/pan state across updates
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                # Simulated User Dot (Blue)
                fig.add_trace(go.Scatter(
                    x=[user_pos[0]], y=[user_pos[1]],
                    mode='markers',
                    marker=dict(color='blue', size=15, symbol='circle', line=dict(color='white', width=2)),
                    name='You (Simulated)'
                ))

                # Evacuation Route (Orange) - only if hazard present or explicitly computing
                if core.HAZARDS and len(path_nodes) > 1:
                    fig.add_trace(go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines',
                        line=dict(color='orange', width=5, dash='dot'),
                        name='Evac Route'
                    ))

                plot.figure = fig
                plot.update()

            # Update loop for public view (simpler/slower than admin)
            ui.timer(1.0, update_public_view)

# gui/actions/__init__.py

from .common import (
    get_workspace_choices,
    smart_resolve_inputs,
    toggle_output_input,
    reset_morph_ui,
    ui_toggle_enabled,
    ui_toggle_disabled,
    ui_toggle_ties,
    resolve_path_priority
)

from .resources import (
    format_resource_html,
    update_resources,
    update_mini_queue,
    refresh_queue_table,
    clear_queue_history
)

from .workspace import (
    refresh_workspace_ui,
    load_files_to_workspace,
    load_from_server_path,
    save_workspace_model,
    delete_workspace_model,
    handle_sidebar_select,
    load_settings,
    save_settings
)

from .analysis import (
    run_analysis,
    apply_recommendation
)

from .submission import (
    submit_extract,
    submit_resize,
    submit_morph,
    submit_merge,
    submit_utils,
    add_files_to_merge,
    add_workspace_to_merge,
    clear_merge_list,
    normalize_weights_ui,
    distribute_weights_ui,
    invert_weights_ui,
    submit_bridge
)

from .checkpoint_merge import submit_checkpoint_merge # Added
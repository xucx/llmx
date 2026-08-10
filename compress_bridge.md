Earlier parts of this conversation have been compressed into a snapshot. The user messages in the compressed part were:

{{ range .ComposedUserPrompts}}
- {{ . }}
{{ end }}

Below is a structured snapshot of the compressed conversation. It is your ONLY memory of that part: rely on it to continue serving the user seamlessly. Do not re-ask for information the user already provided, stay consistent with answers already given, and treat time-sensitive values recorded in the snapshot as historical facts rather than current state.

{{ .Summary }}

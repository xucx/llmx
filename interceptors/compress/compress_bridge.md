Earlier parts of this conversation have been compressed into a snapshot. The latest user messages in the compressed part were (older ones are covered by the snapshot):

{{ range .ComposedUserPrompts}}
- {{ . }}
{{ end }}
{{ if .InProgress }}
The last of these user messages is still in progress: the messages after this snapshot continue its work. Pick up from where they leave off instead of starting over.
{{ end }}
Below is a structured snapshot of the compressed conversation. It is your ONLY memory of that part: rely on it to continue serving the user seamlessly. Do not re-ask for information the user already provided, stay consistent with answers already given, and treat time-sensitive values recorded in the snapshot as historical facts rather than current state.

{{ .Summary }}

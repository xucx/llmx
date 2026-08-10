package utils

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

func JsonSchemaResolve[T any]() (*jsonschema.Resolved, string, error) {
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, "", err
	}
	resolved, err := schema.Resolve(nil)
	if err != nil {
		return nil, "", fmt.Errorf("resolving schema: %w", err)
	}

	desc, err := json.Marshal(resolved.Schema())
	if err != nil {
		return nil, "", err
	}

	return resolved, string(desc), nil
}

func JsonSchemaValidate(resolved *jsonschema.Resolved, str string, result any) error {

	str = strings.TrimPrefix(str, "```json")
	str = strings.TrimSuffix(str, "```")

	var instant any
	if err := json.Unmarshal([]byte(str), &instant); err != nil {
		return fmt.Errorf("unmarshal JSON: %w", err)
	}
	if err := resolved.Validate(instant); err != nil {
		return fmt.Errorf("validate JSON: %w", err)
	}

	return json.Unmarshal([]byte(str), result)
}

package utils

func BoolPtr(v bool) *bool {
	return &v
}

func Ptr[T any](v T) *T {
	return &v
}

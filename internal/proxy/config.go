package proxy

import (
	"os"

	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx"

	"go.yaml.in/yaml/v3"
)

var (
	// default server config
	C Config = Config{
		Log: logx.Config{
			Level: "info",
		},
		Host: "0.0.0.0:9000",
	}
)

type Config struct {
	Log    logx.Config       `yaml:"log"`
	Host   string            `yaml:"host"`
	Tokens []string          `yaml:"tokens"`
	LLM    llmx.ModelsConfig `yaml:"llm"`
}

func LoadConfig(f string) error {
	fd, err := os.ReadFile(f)
	if err != nil {
		return err
	}

	if err := yaml.Unmarshal(fd, &C); err != nil {
		return err
	}

	return nil
}

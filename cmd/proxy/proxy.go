package proxy

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/internal/proxy"
)

var (
	configFile string
	Cmd        = &cobra.Command{
		Use:   "proxy",
		Short: "proxy",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runProxy(cmd.Context())
		},
	}
)

func runProxy(ctx context.Context) error {

	if configFile != "" {
		if err := proxy.LoadConfig(configFile); err != nil {
			return fmt.Errorf("load config file %s: %w", configFile, err)
		}
	}

	logx.SetLogger(logx.NewZapLogger(proxy.C.Log))

	return proxy.Run(ctx)
}

func init() {
	flags := Cmd.Flags()
	flags.StringVarP(&configFile, "config", "c", "", "config file")
}

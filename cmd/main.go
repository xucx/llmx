package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/xucx/llmx/cmd/proxy"
	"github.com/xucx/llmx/internal/version"

	"github.com/spf13/cobra"
)

var (
	configFile string
	cmd        = &cobra.Command{
		Use:           "llmx",
		Short:         "llmx",
		Version:       version.Version + " " + version.BuildRevision + " " + version.BuildTimestamp,
		SilenceErrors: true,
		SilenceUsage:  true,
	}
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	cobra.CheckErr(cmd.ExecuteContext(ctx))
}

func init() {
	cmd.AddCommand(proxy.Cmd)

}

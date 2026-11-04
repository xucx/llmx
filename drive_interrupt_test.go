package llmx

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
)

func TestWatchInterruptReadsAlreadyAppendedInput(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session := NewMemorySession(NewInputEvent(&Input{Mode: InputInterrupt}))
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		stop := watchInterrupt(ctx, session, 0, cancel)
		defer stop()
		synctest.Wait()
		if !errors.Is(context.Cause(ctx), ErrToolInterrupted) {
			t.Fatalf("interrupt appended before watcher started was missed: %v", context.Cause(ctx))
		}
	})
}

func TestWatchInterruptOnlyCancelsForNewInterrupt(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session := NewMemorySession(NewInputEvent(&Input{Mode: InputInterrupt}))
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		stop := watchInterrupt(ctx, session, 1, cancel)
		defer stop()
		synctest.Wait()
		if ctx.Err() != nil {
			t.Fatal("old interrupt canceled current execution")
		}
		for _, mode := range []InputMode{InputInterject, InputQueue, InputInterrupt} {
			if err := session.Append(t.Context(), -1, NewInputEvent(NewUserInput(mode, "next"))); err != nil {
				t.Fatal(err)
			}
			synctest.Wait()
			if mode == InputInterrupt {
				if !errors.Is(context.Cause(ctx), ErrToolInterrupted) {
					t.Fatalf("new interrupt did not cancel execution: %v", context.Cause(ctx))
				}
			} else if ctx.Err() != nil {
				t.Fatalf("%s input canceled execution", mode)
			}
		}
	})
}

func TestWatchInterruptStopsWatching(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session := NewMemorySession()
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		stop := watchInterrupt(ctx, session, 0, cancel)
		synctest.Wait()
		stop()
		synctest.Wait()
		if err := session.Append(t.Context(), -1, NewInputEvent(&Input{Mode: InputInterrupt})); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		if ctx.Err() != nil {
			t.Fatal("stopped watcher canceled execution")
		}
	})
}

type canceledLoadSession struct {
	*MemorySession
	loads int
}

func (s *canceledLoadSession) Load(ctx context.Context) ([]*Event, int64, error) {
	s.loads++
	return nil, 0, ctx.Err()
}

func TestWatchInterruptCanceledLoadDoesNotSpin(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session := &canceledLoadSession{MemorySession: NewMemorySession()}
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		stop := watchInterrupt(ctx, session, 0, cancel)
		defer stop()
		synctest.Wait()
		if session.loads > 1 {
			t.Fatalf("canceled watcher keeps loading: %d", session.loads)
		}
	})
}

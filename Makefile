
PROJECT = llmx
TIME = $(shell date '+%Y/%m/%d-%H:%M:%S')
GIT_COMMIT := ${shell git rev-parse --short HEAD}
ifeq ($(VERSION),)
	CHANGES = $(shell git status --porcelain --untracked-files=no)
	ifneq ($(CHANGES),)
	    DIRTY = -dirty
	endif

	VERSION = $(GIT_COMMIT)$(DIRTY)

	GIT_TAG = $(shell git tag -l --contains HEAD | head -n 1)
	# Override VERSION with the Git tag if the current HEAD has a tag pointing to
	# it AND the worktree isn't dirty.
	ifneq ($(GIT_TAG),)
	    ifeq ($(DIRTY),)
	        VERSION = $(GIT_TAG)
	    endif
	endif
endif

.PHONY: proto
proto:
	buf generate --path api

DIST = $(CURDIR)/dist
EXE = ${PROJECT}
BUILD_OPTS = -ldflags="-s -w -X github.com/xucx/llmx/internal/version.Version=$(VERSION) \
	-X github.com/xucx/llmx/internal/version.BuildRevision=$(GIT_COMMIT) \
	-X github.com/xucx/llmx/internal/version.BuildTimestamp=$(TIME)"

.PHONY: build
build: 
	go build -v -o ${DIST}/${EXE} ${BUILD_OPTS} $(CURDIR)/cmd

PLATFORMS := linux/amd64 windows/amd64 darwin/amd64
.PHONY: build_all $(PLATFORMS)
build_all: $(PLATFORMS)
$(PLATFORMS):
	$(eval os := $(word 1,$(subst /, ,$@)))
	$(eval arch := $(word 2,$(subst /, ,$@)))
	$(eval suffix := $(if $(filter windows,$(os)),.exe))

	GOOS=$(os) GOARCH=$(arch) go build -o ${DIST}/${EXE}-$(os)-$(arch)${suffix} ${BUILD_OPTS} $(CURDIR)/cmd


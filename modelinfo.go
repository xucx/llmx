package llmx

import "strings"

type ModelInfo struct {
	MaxToken       int64 `yaml:"maxToken"`
	MaxOutputToken int64 `yaml:"maxOutputToken"`
}

var DefaultModelInfos = map[string]ModelInfo{
	"gpt-5.4": {
		MaxToken:       1000000, //1M
		MaxOutputToken: 128000,  //128K
	},
	"gpt-5.4-mini": {
		MaxToken:       400000, //400K
		MaxOutputToken: 128000, //128K
	},
	"gpt-5.4-nano": {
		MaxToken:       400000, //400K
		MaxOutputToken: 128000, //128K
	},
	"gemini-3.1-pro-preview": {
		MaxToken:       1064000, //1M + 64K
		MaxOutputToken: 64000,   //64K
	},
	"gemini-3-flash-preview": {
		MaxToken:       1064000, //1M + 64K
		MaxOutputToken: 64000,   //64K
	},
	"gemini-3-pro-image-preview": {
		MaxToken:       97000, //65k + 32K
		MaxOutputToken: 32000, //32k
	},
	"gemini-3.1-flash-image-preview": {
		MaxToken:       160000, //128k + 32K
		MaxOutputToken: 32000,  //32k
	},
	"glm-5": {
		MaxToken:       200000, //200K
		MaxOutputToken: 128000, //128K
	},
	"glm-5.1": {
		MaxToken:       200000, //200K
		MaxOutputToken: 128000, //128K
	},
	"deepseek-chat": { //3.2
		MaxToken:       128000, //128K
		MaxOutputToken: 8000,   //8K
	},
	"deepseek-reasoner": { //3.2
		MaxToken:       128000, //128K
		MaxOutputToken: 64000,  //64K
	},
}

func defaultModelInfo(name string) *ModelInfo {
	splitNames := strings.Split(name, "/")
	if len(splitNames) > 0 {
		if info, ok := DefaultModelInfos[splitNames[len(splitNames)-1]]; ok {
			return &info
		}
	}
	return &ModelInfo{}
}

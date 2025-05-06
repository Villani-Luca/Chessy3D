-- public.moves definition

-- Drop table

-- DROP TABLE public.moves;

-- public.games definition

-- Drop table

-- DROP TABLE public.games;

CREATE TABLE public.games (
	id serial4 NOT NULL,
	"event" varchar(30000) NULL,
	site varchar(30000) NULL,
	"date" varchar(30000) NULL,
	round varchar(30000) NULL,
	white varchar(30000) NULL,
	black varchar(30000) NULL,
	"result" varchar(30000) NULL,
	resultdecimal varchar(30000) NULL,
	whitetitle varchar(30000) NULL,
	blacktitle varchar(30000) NULL,
	whiteelo varchar(30000) NULL,
	blackelo varchar(30000) NULL,
	eco varchar(30000) NULL,
	opening varchar(30000) NULL,
	variation varchar(30000) NULL,
	whitefideid varchar(30000) NULL,
	blackfideid varchar(30000) NULL,
	eventdate varchar(30000) NULL,
	annotator varchar(30000) NULL,
	plycount varchar(30000) NULL,
	timecontrol varchar(30000) NULL,
	"time" varchar(30000) NULL,
	termination varchar(30000) NULL,
	"mode" varchar(30000) NULL,
	fen varchar(30000) NULL,
	setup varchar(30000) NULL,
	moves varchar(30000) NULL,
	"source" varchar(50) NULL,
	CONSTRAINT games_pkey PRIMARY KEY (id)
);

CREATE TABLE public.moves (
	embeddingid bpchar(20) NOT NULL,
	gameid int4 NOT NULL,
	CONSTRAINT moves_pk PRIMARY KEY (embeddingid, gameid)
);
CREATE INDEX moves_embeddingid_idx ON public.moves USING btree (embeddingid);
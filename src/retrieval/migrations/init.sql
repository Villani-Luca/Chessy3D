-- games definition

-- games definition

CREATE TABLE "games"
(
    event         text,
    site          text,
    date          text,
    round         text,
    white         text,
    black         text,
    "result"        text,
    resultdecimal text,
    whitetitle    text,
    blacktitle    text,
    whiteelo      text,
    blackelo      text,
    eco           text,
    opening       text,
    variation     text,
    whitefideid   text,
    blackfideid   text,
    eventdate     text,
    annotator     text,
    plycount      text,
    timecontrol   text,
    time          text,
    termination   text,
    mode          text,
    fen           text,
    setup         text,
    moves         text,
    id            integer
        constraint table_name_pk
            primary key autoincrement,
    "source" text
);

create table if not exists moves
(
    chromaid INTEGER,
    gameid   INTEGER
        CONSTRAINT moves_games_id_fk
            REFERENCES games (id),
    CONSTRAINT moves_pk
        PRIMARY KEY (chromaid, gameid)
);


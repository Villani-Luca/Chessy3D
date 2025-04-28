create table if not exists games
(
    event TEXT,
    site  TEXT,
    date  INTEGER,
    id    INTEGER
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


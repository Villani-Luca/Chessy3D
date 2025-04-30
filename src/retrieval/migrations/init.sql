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

create table if not exists tempgames
(
    Event         text,
    Site          text,
    Date          text,
    Round         text,
    White         text,
    Black         text,
    Result        text,
    ResultDecimal text,
    WhiteTitle    text,
    BlackTitle    text,
    WhiteElo      text,
    BlackElo      text,
    ECO           text,
    Opening       text,
    Variation     text,
    WhiteFideId   text,
    BlackFideId   text,
    EventDate     text,
    Annotator     text,
    PlyCount      text,
    TimeControl   text,
    Time          text,
    Termination   text,
    Mode          text,
    FEN           text,
    SetUp         text,
    Moves         text,
    id            integer
        constraint table_name_pk
            primary key autoincrement
);


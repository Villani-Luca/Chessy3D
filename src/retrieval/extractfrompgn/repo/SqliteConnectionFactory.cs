using Microsoft.Data.Sqlite;

namespace prep.repo;

internal class SqliteConnectionFactory
{
    public static SqliteConnection CreateConnection(string connectionString)
    {
        var conn = new SqliteConnection(connectionString);
        conn.Open();

        var walcommand = conn.CreateCommand();
        walcommand.CommandText = @"PRAGMA journal_mode=WAL;";
        walcommand.ExecuteNonQuery();

        return conn;
    }
}

public class SqliteAddParseResultJITCommand : IDisposable
{
    private readonly SqliteCommand _command;

    public SqliteTransaction? Transaction { get => _command.Transaction; set => _command.Transaction = value; }

    private SqliteAddParseResultJITCommand(SqliteCommand command)
    {
        _command = command;
    }

    public static SqliteAddParseResultJITCommand Create(SqliteConnection connection, SqliteTransaction? transaction = null)
    {
        var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandType = System.Data.CommandType.Text;

        command.CommandText = """
        insert into games (
            event, 
            site, 
            "date", 
            round, 
            white, 
            black, 
            result, 
            resultdecimal, 
            whitetitle, 
            blacktitle,
            whiteelo, 
            blackelo, 
            eco, 
            opening, 
            variation, 
            whitefideid, 
            blackfideid, 
            eventdate, 
            annotator,
            plycount, 
            timecontrol, 
            "time", 
            termination, 
            mode, 
            fen, 
            setup, 
            moves,
            source
        )
        values (
            @Event, 
            @Site, 
            @Date, 
            @Round, 
            @White, 
            @Black, 
            @Result, 
            @ResultDecimal, 
            @WhiteTitle, 
            @BlackTitle,
            @WhiteElo, 
            @BlackElo, 
            @ECO, 
            @Opening, 
            @Variation, 
            @WhiteFideId, 
            @BlackFideId, 
            @EventDate, 
            @Annotator,
            @PlyCount, 
            @TimeControl, 
            @Time, 
            @Termination, 
            @Mode, 
            @FEN, 
            @SetUp, 
            @Moves,
            @Source
        );
        """;

        var parameter_1 = command.CreateParameter();
        var parameter_2 = command.CreateParameter();
        var parameter_3 = command.CreateParameter();
        var parameter_4 = command.CreateParameter();
        var parameter_5 = command.CreateParameter();
        var parameter_6 = command.CreateParameter();
        var parameter_7 = command.CreateParameter();
        var parameter_8 = command.CreateParameter();
        var parameter_9 = command.CreateParameter();
        var parameter_10 = command.CreateParameter();
        var parameter_11 = command.CreateParameter();
        var parameter_12 = command.CreateParameter();
        var parameter_13 = command.CreateParameter();
        var parameter_14 = command.CreateParameter();
        var parameter_15 = command.CreateParameter();
        var parameter_16 = command.CreateParameter();
        var parameter_17 = command.CreateParameter();
        var parameter_18 = command.CreateParameter();
        var parameter_19 = command.CreateParameter();
        var parameter_20 = command.CreateParameter();
        var parameter_21 = command.CreateParameter();
        var parameter_22 = command.CreateParameter();
        var parameter_23 = command.CreateParameter();
        var parameter_24 = command.CreateParameter();
        var parameter_25 = command.CreateParameter();
        var parameter_26 = command.CreateParameter();
        var parameter_27 = command.CreateParameter();
        var parameter_28 = command.CreateParameter();

        parameter_1.ParameterName = "@Event";
        parameter_2.ParameterName = "@Site";
        parameter_3.ParameterName = "@Date";
        parameter_4.ParameterName = "@Round";
        parameter_5.ParameterName = "@White";
        parameter_6.ParameterName = "@Black";
        parameter_7.ParameterName = "@Result";
        parameter_8.ParameterName = "@ResultDecimal";
        parameter_9.ParameterName = "@WhiteTitle";
        parameter_10.ParameterName = "@BlackTitle";
        parameter_11.ParameterName = "@WhiteElo";
        parameter_12.ParameterName = "@BlackElo";
        parameter_13.ParameterName = "@ECO";
        parameter_14.ParameterName = "@Opening";
        parameter_15.ParameterName = "@Variation";
        parameter_16.ParameterName = "@WhiteFideId";
        parameter_17.ParameterName = "@BlackFideId";
        parameter_18.ParameterName = "@EventDate";
        parameter_19.ParameterName = "@Annotator";
        parameter_20.ParameterName = "@PlyCount";
        parameter_21.ParameterName = "@TimeControl";
        parameter_22.ParameterName = "@Time";
        parameter_23.ParameterName = "@Termination";
        parameter_24.ParameterName = "@Mode";
        parameter_25.ParameterName = "@FEN";
        parameter_26.ParameterName = "@SetUp";
        parameter_27.ParameterName = "@Moves";
        parameter_28.ParameterName = "@Source";

        command.Parameters.Add(parameter_1);
        command.Parameters.Add(parameter_2);
        command.Parameters.Add(parameter_3);
        command.Parameters.Add(parameter_4);
        command.Parameters.Add(parameter_5);
        command.Parameters.Add(parameter_6);
        command.Parameters.Add(parameter_7);
        command.Parameters.Add(parameter_8);
        command.Parameters.Add(parameter_9);
        command.Parameters.Add(parameter_10);
        command.Parameters.Add(parameter_11);
        command.Parameters.Add(parameter_12);
        command.Parameters.Add(parameter_13);
        command.Parameters.Add(parameter_14);
        command.Parameters.Add(parameter_15);
        command.Parameters.Add(parameter_16);
        command.Parameters.Add(parameter_17);
        command.Parameters.Add(parameter_18);
        command.Parameters.Add(parameter_19);
        command.Parameters.Add(parameter_20);
        command.Parameters.Add(parameter_21);
        command.Parameters.Add(parameter_22);
        command.Parameters.Add(parameter_23);
        command.Parameters.Add(parameter_24);
        command.Parameters.Add(parameter_25);
        command.Parameters.Add(parameter_26);
        command.Parameters.Add(parameter_27);
        command.Parameters.Add(parameter_28);

        return new SqliteAddParseResultJITCommand(command);
    }

    public void Execute(ParseResultJIT result)
    {
        _command.Parameters[0].Value    = string.IsNullOrWhiteSpace(result.Event) ? DBNull.Value : result.Event;
        _command.Parameters[1].Value    = string.IsNullOrWhiteSpace(result.Site) ? DBNull.Value : result.Site;
        _command.Parameters[2].Value    = string.IsNullOrWhiteSpace(result.Date) ? DBNull.Value : result.Date;
        _command.Parameters[3].Value    = string.IsNullOrWhiteSpace(result.Round) ? DBNull.Value : result.Round;
        _command.Parameters[4].Value    = string.IsNullOrWhiteSpace(result.White) ? DBNull.Value : result.White;
        _command.Parameters[5].Value    = string.IsNullOrWhiteSpace(result.Black) ? DBNull.Value : result.Black;
        _command.Parameters[6].Value    = string.IsNullOrWhiteSpace(result.Result) ? DBNull.Value : result.Result;
        _command.Parameters[7].Value    = string.IsNullOrWhiteSpace(result.ResultDecimal) ? DBNull.Value : result.ResultDecimal;
        _command.Parameters[8].Value    = string.IsNullOrWhiteSpace(result.WhiteTitle) ? DBNull.Value : result.WhiteTitle;
        _command.Parameters[9].Value    = string.IsNullOrWhiteSpace(result.BlackTitle) ? DBNull.Value : result.BlackTitle;
        _command.Parameters[10].Value   = string.IsNullOrWhiteSpace(result.WhiteElo) ? DBNull.Value : result.WhiteElo;
        _command.Parameters[11].Value   = string.IsNullOrWhiteSpace(result.BlackElo) ? DBNull.Value : result.BlackElo;
        _command.Parameters[12].Value   = string.IsNullOrWhiteSpace(result.ECO) ? DBNull.Value : result.ECO;
        _command.Parameters[13].Value   = string.IsNullOrWhiteSpace(result.Opening) ? DBNull.Value : result.Opening;
        _command.Parameters[14].Value   = string.IsNullOrWhiteSpace(result.Variation) ? DBNull.Value : result.Variation;
        _command.Parameters[15].Value   = string.IsNullOrWhiteSpace(result.WhiteFideId) ? DBNull.Value : result.WhiteFideId;
        _command.Parameters[16].Value   = string.IsNullOrWhiteSpace(result.BlackFideId) ? DBNull.Value : result.BlackFideId;
        _command.Parameters[17].Value   = string.IsNullOrWhiteSpace(result.EventDate) ? DBNull.Value : result.EventDate;
        _command.Parameters[18].Value   = string.IsNullOrWhiteSpace(result.Annotator) ? DBNull.Value : result.Annotator;
        _command.Parameters[19].Value   = string.IsNullOrWhiteSpace(result.PlyCount) ? DBNull.Value : result.PlyCount;
        _command.Parameters[20].Value   = string.IsNullOrWhiteSpace(result.TimeControl) ? DBNull.Value : result.TimeControl;
        _command.Parameters[21].Value   = string.IsNullOrWhiteSpace(result.Time) ? DBNull.Value : result.Time;
        _command.Parameters[22].Value   = string.IsNullOrWhiteSpace(result.Termination) ? DBNull.Value : result.Termination;
        _command.Parameters[23].Value   = string.IsNullOrWhiteSpace(result.Mode) ? DBNull.Value : result.Mode;
        _command.Parameters[24].Value   = DBNull.Value;
        _command.Parameters[25].Value   = DBNull.Value;
        _command.Parameters[26].Value   = string.IsNullOrWhiteSpace(result.Moves) ? DBNull.Value : result.Moves;
        _command.Parameters[27].Value   = string.IsNullOrWhiteSpace(result.Source) ? DBNull.Value : result.Source;

        _command.ExecuteNonQuery();
    }

    public void Dispose()
    {
        _command.Dispose();
    }
}

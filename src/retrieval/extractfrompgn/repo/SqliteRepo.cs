using Microsoft.Data.Sqlite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace prep.repo;

internal class SqliteRepo
{
    private readonly SqliteConnection _connection;

    public SqliteRepo(SqliteConnection connection)
    {
        _connection = connection;
    }

    public void ClearTempTable()
    {
        using var command = _connection.CreateCommand();
        command.CommandType = System.Data.CommandType.Text;
        command.CommandText = "delete from games;";
        command.ExecuteNonQuery();
    }

    /*
    public void Save(ParseResult result)
    {
        using var transaction = _connection.BeginTransaction();
        using var command = _connection.CreateCommand();
        try
        {
            command.Transaction = transaction;
            command.CommandType = System.Data.CommandType.Text;
            command.CommandText = """
            insert into tempgames (
                Event, 
                Site, 
                "Date", 
                Round, 
                White, 
                Black, 
                Result, 
                ResultDecimal, 
                WhiteTitle, 
                BlackTitle,
                WhiteElo, 
                BlackElo, 
                ECO, 
                Opening, 
                Variation, 
                WhiteFideId, 
                BlackFideId, 
                EventDate, 
                Annotator,
                PlyCount, 
                TimeControl, 
                "Time", 
                Termination, 
                Mode, 
                FEN, 
                SetUp, 
                Moves
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
                @Moves
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

            for(int i = 0; i < result.Parsed; i++) 
            {
                parameter_1.Value  = string.IsNullOrWhiteSpace(result.Event[i]) ? DBNull.Value : result.Event[i];
                parameter_2.Value  = string.IsNullOrWhiteSpace(result.Site[i]) ? DBNull.Value : result.Site[i];
                parameter_3.Value  = string.IsNullOrWhiteSpace(result.Date[i]) ? DBNull.Value : result.Date[i];
                parameter_4.Value  = string.IsNullOrWhiteSpace(result.Round[i]) ? DBNull.Value : result.Round[i];
                parameter_5.Value  = string.IsNullOrWhiteSpace(result.White[i]) ? DBNull.Value : result.White[i];
                parameter_6.Value  = string.IsNullOrWhiteSpace(result.Black[i]) ? DBNull.Value : result.Black[i];
                parameter_7.Value  = string.IsNullOrWhiteSpace(result.Result[i]) ? DBNull.Value : result.Result[i];
                parameter_8.Value  = string.IsNullOrWhiteSpace(result.ResultDecimal[i]) ? DBNull.Value : result.ResultDecimal[i];
                parameter_9.Value =  string.IsNullOrWhiteSpace(result.WhiteTitle[i]) ? DBNull.Value : result.WhiteTitle[i];
                parameter_10.Value = string.IsNullOrWhiteSpace(result.BlackTitle[i]) ? DBNull.Value : result.BlackTitle[i];
                parameter_11.Value = string.IsNullOrWhiteSpace(result.WhiteElo[i]) ? DBNull.Value : result.WhiteElo[i];
                parameter_12.Value = string.IsNullOrWhiteSpace(result.BlackElo[i]) ? DBNull.Value : result.BlackElo[i];
                parameter_13.Value = string.IsNullOrWhiteSpace(result.ECO[i]) ? DBNull.Value : result.ECO[i];
                parameter_14.Value = string.IsNullOrWhiteSpace(result.Opening[i]) ? DBNull.Value : result.Opening[i];
                parameter_15.Value = string.IsNullOrWhiteSpace(result.Variation[i]) ? DBNull.Value : result.Variation[i];
                parameter_16.Value = string.IsNullOrWhiteSpace(result.WhiteFideId[i]) ? DBNull.Value : result.WhiteFideId[i];
                parameter_17.Value = string.IsNullOrWhiteSpace(result.BlackFideId[i]) ? DBNull.Value : result.BlackFideId[i];
                parameter_18.Value = string.IsNullOrWhiteSpace(result.EventDate[i]) ? DBNull.Value : result.EventDate[i];
                parameter_19.Value = string.IsNullOrWhiteSpace(result.Annotator[i]) ? DBNull.Value : result.Annotator[i];
                parameter_20.Value = string.IsNullOrWhiteSpace(result.PlyCount[i]) ? DBNull.Value : result.PlyCount[i];
                parameter_21.Value = string.IsNullOrWhiteSpace(result.TimeControl[i]) ? DBNull.Value : result.TimeControl[i];
                parameter_22.Value = string.IsNullOrWhiteSpace(result.Time[i]) ? DBNull.Value : result.Time[i];
                parameter_23.Value = string.IsNullOrWhiteSpace(result.Termination[i]) ? DBNull.Value : result.Termination[i];
                parameter_24.Value = string.IsNullOrWhiteSpace(result.Mode[i]) ? DBNull.Value : result.Mode[i];
                parameter_25.Value = DBNull.Value;
                parameter_26.Value = DBNull.Value;
                parameter_27.Value = string.IsNullOrWhiteSpace(result.Moves[i]) ? DBNull.Value : result.Moves[i];

                command.ExecuteNonQuery();
            } 
            transaction.Commit();
        }
        catch (Exception ex)
        {
            transaction.Rollback();
            throw;
        }
    }
    */
}

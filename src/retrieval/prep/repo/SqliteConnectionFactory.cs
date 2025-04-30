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

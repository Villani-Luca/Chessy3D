using Microsoft.Data.Sqlite;
using prep;
using prep.repo;
using Serilog;
using System.Collections.Concurrent;


public record ConsumerThreadParams (
    int ThreadId,
    ConcurrentQueue<ParseResultJIT> Queue, 
    CancellationToken Token,
    string ConnectionString
);

internal class ParseResultConsumer : IDisposable
{
    private readonly int _threadId;
    private readonly ConcurrentQueue<ParseResultJIT> _queue;
    private readonly CancellationToken _token;
    private readonly SqliteConnection _sqliteConnection;
    private readonly SqliteAddParseResultJITCommand _command;
    //private readonly SqliteRepo _sqliteRepo;

    public void Dispose()
    {
        _command.Dispose();
        _sqliteConnection.Dispose();
    }

    public static void ThreadConsumerFunc(object? data)
    {
        if (data is not ConsumerThreadParams p)
            return;

        var connection = SqliteConnectionFactory.CreateConnection(p.ConnectionString);
        var consumer = new ParseResultConsumer(p.Queue, p.Token, connection, p.ThreadId);
        consumer.Consume();
    }

    public ParseResultConsumer(
        ConcurrentQueue<ParseResultJIT> queue, 
        CancellationToken token,
        SqliteConnection conn,
        int threadId = 0
    //SqliteRepo repo
    )
    {
        // Initialize the consumer
        _queue = queue;
        _token = token;
        _sqliteConnection = conn;

        _command = SqliteAddParseResultJITCommand.Create(_sqliteConnection);
        _threadId = threadId;
     }

    private SqliteTransaction EnsureTransaction()
    {
        while (true)
        {
            try
            {
                return _sqliteConnection.BeginTransaction();
            }
            catch(SqliteException ex)
            {
                Log.Error(ex, "Error creating transaction: {message}", ex.Message);
                Thread.Sleep(100);
            }
        }
    }

    public void Consume()
    {
        long processed = 0;


        var transaction = EnsureTransaction();
        _command.Transaction = transaction;

        while (true)
        {
            if (!_queue.TryDequeue(out var result))
            {
                if (_token.IsCancellationRequested)
                    break;
                

                Thread.Sleep(10);
                continue;
            }

            _command.Execute(result);
            processed += 1;

            if (processed % 5000 == 0)
            {
                transaction.Commit();
                transaction = EnsureTransaction();
                _command.Transaction = transaction;

                Log.Information("[{thread} - SAVED] {processed}", _threadId, processed);
            }
        }

        transaction.Commit();
        _command.Dispose();
        transaction.Dispose();
    }
}

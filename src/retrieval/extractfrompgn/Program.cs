using Cocona;
using Microsoft.Data.Sqlite;
using prep;
using prep.repo;
using SQLitePCL;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Serilog;

// See https://aka.ms/new-console-template for more information
Log.Logger = new LoggerConfiguration()
    .WriteTo.Async(a => a.Console())
    .WriteTo.Async(a => a.File("logs/log.log"))
    .CreateLogger();

CoconaLiteApp.Run((
    string pgnfolder = """D:\Projects\Uni\Chessy3D\data\retrieval\lumbrasgigabase\splitted""", 
    string chromaurl = "http://localhost:8000", 
    string sqliteconn = """Host=localhost;Username=postgres;Password=password;Database=chessy""",
    bool clear_db = true,
    int max_file_record_size = 5000
    ) =>
{
    // ##### Services primitive DI #####

    //using HttpClient httpClient = new HttpClient();

    /*
    if(clear_db)
    {
        using var sqliteConnection = SqliteConnectionFactory.CreateConnection(sqliteconn);
        SqliteRepo sqliteRepo = new SqliteRepo(sqliteConnection);
        sqliteRepo.ClearTempTable();
        Log.Information("Cleared database.");
    }
    */

    var dirinfo = new DirectoryInfo(pgnfolder);
    var filelist = dirinfo.GetFiles("*.pgn", SearchOption.TopDirectoryOnly);
    if(filelist.Length == 0)
    {
        Log.Error("No PGN files found in the specified directory.");
        return;
    }

    var ordered = filelist.OrderBy(x => x.LastWriteTime).ToList();

    var queue = new ConcurrentQueue<ParseResultJIT>();
    var cancellationTokenSource = new CancellationTokenSource();

    const int consumer_thread_number = 5;
    Thread[] threadArray = new Thread[consumer_thread_number];
    for(int i = 0; i < consumer_thread_number; i++)
    {
        threadArray[i] = new Thread(ParseResultConsumer.ThreadConsumerFunc);
        threadArray[i].Start(new ConsumerThreadParams(i, queue, cancellationTokenSource.Token, sqliteconn));
    }

    var options = new ParallelOptions
    {
        MaxDegreeOfParallelism = 20,
    };
    Parallel.ForEach(ordered, options, (pgnfile, _, batch) =>
    {
        string pgnfileContent = File.ReadAllText(pgnfile.FullName);
        PgnParserJIT.Parse(pgnfileContent, queue);

        Log.Information("[Parsed] - file " + pgnfile.Name + " - " + batch);
    });

    cancellationTokenSource.Cancel();

    foreach (var thread in threadArray)
        thread.Join();
});


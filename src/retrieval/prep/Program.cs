using Cocona;
using Microsoft.Data.Sqlite;
using prep;
using prep.repo;
using System.Runtime.CompilerServices;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");


CoconaLiteApp.Run((
    string pgnfolder = """D:\Projects\Uni\Chessy3D\data\retrieval\lumbrasgigabase\splitted""", 
    string chromaurl = "http://localhost:8000", 
    string sqliteconn = """Data Source=D:\Projects\Uni\Chessy3D\data\retrieval\sqlite.db"""
    ) =>
{
    // ##### Services primitive DI #####

    //using HttpClient httpClient = new HttpClient();
    using SqliteConnection sqliteConnection = SqliteConnectionFactory.CreateConnection(sqliteconn);

    //ChromaDB.Client.ChromaConfigurationOptions chromaConfigurationOptions = new ChromaDB.Client.ChromaConfigurationOptions(chromaurl);
    //ChromaDB.Client.ChromaClient client = new ChromaDB.Client.ChromaClient(chromaConfigurationOptions, httpClient);

    //ChromaRepo chromaRepo = new ChromaRepo(client);
    SqliteRepo sqliteRepo = new SqliteRepo(sqliteConnection);

    var pgnfiles = Directory.GetFiles(pgnfolder, "*.pgn", SearchOption.TopDirectoryOnly);
    if(pgnfiles.Length == 0)
    {
        Console.WriteLine("No PGN files found in the specified directory.");
        return;
    }

    // split in batches
    long processed = 0;
    var stopwatch = new System.Diagnostics.Stopwatch();
    foreach (var pgnfile in pgnfiles)
    {
        stopwatch.Start();
        string pgnfileContent = File.ReadAllText(pgnfile);
        var parser = new PgnParser(null!);
        var result = parser.Parse(pgnfileContent, 5000);

        processed += result.Parsed;
        sqliteRepo.Save(result);

        stopwatch.Stop();
        Console.WriteLine($"Parsed {result.Parsed} games from {pgnfile}. TOTAL: {processed} - {stopwatch.Elapsed.TotalMilliseconds}");
    }
});


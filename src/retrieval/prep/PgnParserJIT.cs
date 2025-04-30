using System.Collections.Concurrent;

namespace prep;

public record ParseResultJIT
{
    public string? Event;
    public string? Site;
    public string? Date;
    public string? Round;
    public string? White;
    public string? Black;
    public string? Result;
    public string? ResultDecimal;
    public string? WhiteTitle;
    public string? BlackTitle;
    public string? WhiteElo;
    public string? BlackElo;
    public string? ECO;
    public string? Opening;
    public string? Variation;
    public string? WhiteFideId;
    public string? BlackFideId;
    public string? EventDate;
    public string? Annotator;
    public string? PlyCount;
    public string? TimeControl;
    public string? Time;
    public string? Termination;
    public string? Mode;
    //public string?[] FEN;
    //public string?[] SetUp;
    public string? Moves;
    //public string?[] Embedding;
    public string? Source;

    /*
    public ParseResult(int n)
    {
        Event = new string[n];
        Site = new string[n];
        Date = new string[n];
        Round = new string[n];
        White = new string[n];
        Black = new string[n];
        Result = new string[n];
        ResultDecimal = new string[n];
        WhiteTitle = new string[n];
        BlackTitle = new string[n];
        WhiteElo = new string[n];
        BlackElo = new string[n];
        ECO = new string[n];
        Opening = new string[n];
        Variation = new string[n];
        WhiteFideId = new string[n];
        BlackFideId = new string[n];
        EventDate = new string[n];
        Annotator = new string[n];
        PlyCount = new string[n];
        TimeControl = new string[n];
        Time = new string[n];
        Termination = new string[n];
        Mode = new string[n];
        Moves = new string[n];
        Source = new string[n];
    }
    */

}

internal class PgnParserJIT
{
    public static void Parse(ReadOnlySpan<char> content, ConcurrentQueue<ParseResultJIT> queue)
    {
        ParseResultJIT result = new ParseResultJIT();

        var enumerator = content.EnumerateLines();
        var exit = false;
        do {
            var line = enumerator.Current;
            if (line.StartsWith("["))
            {
                var space = line.IndexOf(' ');
                var header_end = line.IndexOf(']');

                if (line.StartsWith("[Event ")) result.Event = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Site ")) result.Site = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Date ")) result.Date = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Round ")) result.Round = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[White ")) result.White = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Black ")) result.Black = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Result ")) result.Result = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[ResultDecimal ")) result.ResultDecimal = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteTitle ")) result.WhiteTitle = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackTitle ")) result.BlackTitle = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteElo ")) result.WhiteElo = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackElo ")) result.BlackElo = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[ECO ")) result.ECO = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Opening ")) result.Opening = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Variation ")) result.Variation = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteFideId ")) result.WhiteFideId = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackFideId ")) result.BlackFideId = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[EventDate ")) result.EventDate = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[PlyCount ")) result.Annotator = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[TimeControl ")) result.TimeControl = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Time ")) result.Time = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Termination ")) result.Termination = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Mode ")) result.Mode = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[Moves ")) result.Moves = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[FEN ")) result.FEN = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[Setup ")) result.Annotator = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Source ")) result.Source = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
            }
            else if (!line.IsEmpty)
            {
                string moves = "";
                while (!line.IsEmpty)
                {
                    moves += " " + line.ToString().TrimEnd();
                    exit = !enumerator.MoveNext();
                    if (exit)
                        break;

                    line = enumerator.Current;
                }

                result.Moves = moves;

                queue.Enqueue(result); 
            }
        } while (!exit && enumerator.MoveNext());
    }
}

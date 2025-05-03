namespace prep;

public record ParseResult
{
    public int Parsed;

    public string[] Event;
    public string[] Site;
    public string[] Date;
    public string[] Round;
    public string[] White;
    public string[] Black;
    public string[] Result;
    public string[] ResultDecimal;
    public string[] WhiteTitle;
    public string[] BlackTitle;
    public string[] WhiteElo;
    public string[] BlackElo;
    public string[] ECO;
    public string[] Opening;
    public string[] Variation;
    public string[] WhiteFideId;
    public string[] BlackFideId;
    public string[] EventDate;
    public string[] Annotator;
    public string[] PlyCount;
    public string[] TimeControl;
    public string[] Time;
    public string[] Termination;
    public string[] Mode;
    //public string?[] FEN;
    //public string?[] SetUp;
    public string[] Moves;
    //public string?[] Embedding;
    public string[] Source;

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

}

internal class PgnParser
{
    public ParseResult Parse(ReadOnlySpan<char> content, int max)
    {
        var result = new ParseResult(max);

        var currentGame = 0;

        var enumerator = content.EnumerateLines();
        var exit = false;
        do {
            var line = enumerator.Current;
            if (line.StartsWith("["))
            {
                var space = line.IndexOf(' ');
                var header_end = line.IndexOf(']');

                if (line.StartsWith("[Event ")) result.Event[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Site ")) result.Site[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Date ")) result.Date[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Round ")) result.Round[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[White ")) result.White[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Black ")) result.Black[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Result ")) result.Result[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[ResultDecimal ")) result.ResultDecimal[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteTitle ")) result.WhiteTitle[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackTitle ")) result.BlackTitle[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteElo ")) result.WhiteElo[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackElo ")) result.BlackElo[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[ECO ")) result.ECO[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Opening ")) result.Opening[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Variation ")) result.Variation[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[WhiteFideId ")) result.WhiteFideId[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[BlackFideId ")) result.BlackFideId[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[EventDate ")) result.EventDate[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[PlyCount ")) result.Annotator[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[TimeControl ")) result.TimeControl[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Time ")) result.Time[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Termination ")) result.Termination[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Mode ")) result.Mode[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[Moves ")) result.Moves[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[FEN ")) result.FEN[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                // else if (line.StartsWith("[Setup ")) result.Annotator[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
                else if (line.StartsWith("[Source ")) result.Source[currentGame] = line.Slice(space + 1, header_end - space - 1).Trim('"').ToString();
            }
            else if (!line.IsEmpty)
            {
                string moves = "";
                while (!line.IsEmpty)
                {
                    moves += line.ToString().TrimEnd();
                    exit = !enumerator.MoveNext();
                    if (exit)
                        break;

                    line = enumerator.Current;
                }

                result.Moves[currentGame] = moves;
                currentGame += 1;
                // parse moves, read everything till next line 
            }
        } while (!exit && enumerator.MoveNext());

        result.Parsed = currentGame;

        return result;
    }
}

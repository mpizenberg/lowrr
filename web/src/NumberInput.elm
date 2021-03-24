module NumberInput exposing (Field, IntError(..), intDefault, setMaxBound, setMinBound, updateInt)

import Form.Decoder exposing (Decoder)


type alias Field num err =
    { defaultValue : num
    , min : Maybe num
    , max : Maybe num
    , increase : num -> num
    , decrease : num -> num
    , input : String
    , decodedInput : Result (List err) num
    }


intDefault : Field Int IntError
intDefault =
    { defaultValue = 0
    , min = Nothing
    , max = Nothing
    , increase = \n -> n + 1
    , decrease = \n -> n - 1
    , input = "0"
    , decodedInput = Ok 0
    }


setMinBound : Maybe number -> Field number err -> Field number err
setMinBound newMin field =
    { field | min = newMin }


setMaxBound : Maybe number -> Field number err -> Field number err
setMaxBound newMax field =
    { field | max = newMax }


updateInt : String -> Field Int IntError -> Field Int IntError
updateInt input field =
    { field | input = input, decodedInput = Form.Decoder.run (intDecoder field.min field.max) input }


type IntError
    = IntParsingError
    | IntTooSmall { bound : Int, actual : Int }
    | IntTooBig { bound : Int, actual : Int }


intDecoder : Maybe Int -> Maybe Int -> Decoder String IntError Int
intDecoder maybeMin maybeMax =
    Form.Decoder.int IntParsingError
        |> validateMinInt maybeMin
        |> validateMaxInt maybeMax


validateMinInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMinInt maybeMin decoder =
    case maybeMin of
        Nothing ->
            decoder

        Just minInt ->
            Form.Decoder.assert (minBound IntTooSmall minInt) decoder


validateMaxInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMaxInt maybeMax decoder =
    case maybeMax of
        Nothing ->
            decoder

        Just maxInt ->
            Form.Decoder.assert (maxBound IntTooBig maxInt) decoder



-- Helper


minBound : ({ bound : number, actual : number } -> err) -> number -> Decoder number err ()
minBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual < bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )


maxBound : ({ bound : number, actual : number } -> err) -> number -> Decoder number err ()
maxBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual > bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )

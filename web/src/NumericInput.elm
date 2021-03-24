module NumericInput exposing (IntConfig, IntError(..), defaultIntConfig, intDecoder, setIntConfigMax, setIntConfigMin)

import Form.Decoder exposing (Decoder)



-- Integer


type alias IntConfig =
    { defaultValue : Int
    , min : Maybe Int
    , max : Maybe Int
    , increase : Int -> Int
    , decrease : Int -> Int
    }


defaultIntConfig : IntConfig
defaultIntConfig =
    { defaultValue = 0
    , min = Nothing
    , max = Nothing
    , increase = \n -> n + 1
    , decrease = \n -> n - 1
    }


setIntConfigMin : Maybe Int -> IntConfig -> IntConfig
setIntConfigMin newMin config =
    { config | min = newMin }


setIntConfigMax : Maybe Int -> IntConfig -> IntConfig
setIntConfigMax newMax config =
    { config | max = newMax }


type IntError
    = IntParsingError
    | IntTooSmall { bound : Int, actual : Int }
    | IntTooBig { bound : Int, actual : Int }


intDecoder : IntConfig -> Decoder String IntError Int
intDecoder config =
    Form.Decoder.int IntParsingError
        |> validateMinInt config.min
        |> validateMaxInt config.max


validateMinInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMinInt maybeMin decoder =
    case maybeMin of
        Nothing ->
            decoder

        Just minInt ->
            decoder
                |> Form.Decoder.assert (minBound IntTooSmall minInt)


validateMaxInt : Maybe Int -> Decoder input IntError Int -> Decoder input IntError Int
validateMaxInt maybeMax decoder =
    case maybeMax of
        Nothing ->
            decoder

        Just maxInt ->
            decoder
                |> Form.Decoder.assert (maxBound IntTooBig maxInt)



-- Float
-- Helper


minBound : ({ bound : comparable, actual : comparable } -> err) -> comparable -> Decoder comparable err ()
minBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual < bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )


maxBound : ({ bound : comparable, actual : comparable } -> err) -> comparable -> Decoder comparable err ()
maxBound errorTag bound =
    Form.Decoder.custom
        (\actual ->
            if actual > bound then
                Err [ errorTag { bound = bound, actual = actual } ]

            else
                Ok ()
        )

using FAIR.Application.DTOs;
using FluentValidation;

namespace FAIR.Application.Validations
{
    public class ValidationService : IValidationService
    {
        public async Task<ServiceResponse> ValidateAsync<T>(T model, IValidator<T> validator)
        {
            var _validation = await validator.ValidateAsync(model);
            if (!_validation.IsValid)
            {
                var errors = _validation.Errors.Select(e => e.ErrorMessage).ToList();
                string errorsToString = string.Join("; ", errors);
                return new ServiceResponse { message = errorsToString };

            }
            return new ServiceResponse { Success = true };
        }
    }
}
